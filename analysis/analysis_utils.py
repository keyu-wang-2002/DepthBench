from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLMO_CORE_SRC = PROJECT_ROOT / "pretrain" / "OLMo-core" / "src"
if str(OLMO_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(OLMO_CORE_SRC))

from examples.huggingface.convert_checkpoint_to_hf import convert_checkpoint_to_hf  # noqa: E402
from olmo_core.config import DType  # noqa: E402
from olmo_core.data.parquet import load_parquet_tokenizer  # noqa: E402
from olmo_core.data.tokenizer import TokenizerConfig  # noqa: E402
from olmo_core.distributed.checkpoint import load_state_dict  # noqa: E402
from olmo_core.nn.transformer.config import TransformerConfig  # noqa: E402


DEFAULT_SAMPLE_TEXTS = [
    "DepthBench studies how transformer layers contribute to reasoning and language modeling behavior.",
    "A reliable analysis script should load the same model format used elsewhere in the repository.",
    "Residual connections preserve information while attention and feed-forward blocks transform it.",
    "Small synthetic text samples are enough to smoke test metrics before running on a large corpus.",
    "Layer swapping and layer ablation are useful probes for understanding distributed computation.",
]


class IndexedLayerContainer(Sequence[nn.Module]):
    def __init__(self, modules: nn.ModuleDict):
        self.modules = modules

    def _keys(self) -> list[str]:
        return sorted(self.modules.keys(), key=int)

    def __len__(self) -> int:
        return len(self.modules)

    def __iter__(self) -> Iterable[nn.Module]:
        for key in self._keys():
            yield self.modules[key]

    def __getitem__(self, index: int) -> nn.Module:
        return self.modules[self._keys()[index]]

    def __setitem__(self, index: int, value: nn.Module) -> None:
        self.modules[self._keys()[index]] = value


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_device_type(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


def resolve_load_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if get_device_type(device) == "cuda" else torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype]


def autocast_context(device: str, dtype: torch.dtype):
    device_type = get_device_type(device)
    if device_type != "cuda":
        return nullcontext()
    if dtype not in {torch.float16, torch.bfloat16}:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


def _has_hf_weights(path: Path) -> bool:
    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    return any(any(path.glob(pattern)) for pattern in patterns)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def looks_like_hf_model(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists() or not _has_hf_weights(path):
        return False

    config = _load_json(config_path)
    return "model" not in config


def looks_like_olmo_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists():
        return False

    config = _load_json(config_path)
    return isinstance(config, dict) and "model" in config and "dataset" in config


def ensure_hf_model_dir(
    model_path: str,
    output_dir: Path,
    tokenizer_id: str | None,
    conversion_dtype: str,
    max_sequence_length: int | None,
    skip_conversion_validation: bool,
) -> tuple[str, bool]:
    candidate = Path(model_path).expanduser()
    if not candidate.exists():
        return model_path, False

    candidate = candidate.resolve()
    if looks_like_hf_model(candidate):
        return str(candidate), False

    if not looks_like_olmo_checkpoint(candidate):
        raise ValueError(
            f"Unsupported model path: {candidate}. Expected either a Hugging Face model dir or an OLMo checkpoint dir."
        )

    converted_dir = output_dir / "converted_hf_model"
    if looks_like_hf_model(converted_dir):
        return str(converted_dir), True

    output_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = _load_json(candidate / "config.json")
    convert_checkpoint_to_hf(
        original_checkpoint_path=candidate,
        output_path=converted_dir,
        transformer_config_dict=experiment_config["model"],
        tokenizer_config_dict=experiment_config["dataset"]["tokenizer"],
        dtype=DType(conversion_dtype),
        tokenizer_id=tokenizer_id,
        max_sequence_length=max_sequence_length,
        validate=not skip_conversion_validation,
    )
    return str(converted_dir), True


def _mark_analysis_backend(model, backend: str) -> None:
    setattr(model, "_analysis_backend", backend)


def get_model_backend(model) -> str:
    return getattr(model, "_analysis_backend", "hf")


def _load_olmo_core_model_and_tokenizer(
    model_path: str,
    resolved_device: str,
    load_dtype: torch.dtype,
    tokenizer_id: str | None,
    max_sequence_length: int | None,
):
    checkpoint_dir = Path(model_path).expanduser().resolve()
    experiment_config = _load_json(checkpoint_dir / "config.json")

    model_config = TransformerConfig.from_dict(experiment_config["model"])
    tokenizer_config = TokenizerConfig.from_dict(experiment_config["dataset"]["tokenizer"])
    tokenizer_name_or_path = tokenizer_id or tokenizer_config.identifier
    if tokenizer_name_or_path is None:
        raise ValueError("Tokenizer identifier is missing from the checkpoint config and no --tokenizer-id was provided.")

    tokenizer = load_parquet_tokenizer(
        tokenizer_name_or_path,
        tokenizer_config,
        model_max_length=max_sequence_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer_config.pad_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer_config.eos_token_id

    model = model_config.build(init_device="cpu")
    with TemporaryDirectory() as work_dir:
        load_state_dict(
            checkpoint_dir / "model_and_optim",
            {"model": model.state_dict()},
            work_dir=work_dir,
        )

    model.to(device=resolved_device, dtype=load_dtype)
    model.eval()
    _mark_analysis_backend(model, "olmo_core")

    return model, tokenizer, str(checkpoint_dir), False


def _load_hf_model_and_tokenizer(
    model_path: str,
    output_dir: Path,
    resolved_device: str,
    load_dtype: torch.dtype,
    tokenizer_id: str | None,
    max_sequence_length: int | None,
    skip_conversion_validation: bool,
):
    hf_model_path, was_converted = ensure_hf_model_dir(
        model_path=model_path,
        output_dir=output_dir,
        tokenizer_id=tokenizer_id,
        conversion_dtype="bfloat16" if load_dtype == torch.bfloat16 else str(load_dtype).replace("torch.", ""),
        max_sequence_length=max_sequence_length,
        skip_conversion_validation=skip_conversion_validation,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    )
    model.to(resolved_device)
    model.eval()
    _mark_analysis_backend(model, "hf")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, hf_model_path, was_converted


def load_model_and_tokenizer(
    model_path: str,
    output_dir: Path,
    device: str,
    dtype: str,
    tokenizer_id: str | None,
    max_sequence_length: int | None,
    skip_conversion_validation: bool,
    model_backend: str = "auto",
):
    resolved_device = resolve_device(device)
    load_dtype = resolve_load_dtype(dtype, resolved_device)
    candidate = Path(model_path).expanduser()

    if model_backend not in {"auto", "olmo_core", "hf"}:
        raise ValueError(f"Unsupported model backend '{model_backend}'")

    prefer_native = model_backend in {"auto", "olmo_core"}
    if candidate.exists() and looks_like_olmo_checkpoint(candidate.resolve()) and prefer_native:
        model, tokenizer, resolved_model_path, was_converted = _load_olmo_core_model_and_tokenizer(
            model_path=model_path,
            resolved_device=resolved_device,
            load_dtype=load_dtype,
            tokenizer_id=tokenizer_id,
            max_sequence_length=max_sequence_length,
        )
        return model, tokenizer, resolved_device, load_dtype, resolved_model_path, was_converted

    if model_backend == "olmo_core":
        raise ValueError(f"Model path '{model_path}' is not an OLMo-core checkpoint directory.")

    model, tokenizer, resolved_model_path, was_converted = _load_hf_model_and_tokenizer(
        model_path=model_path,
        output_dir=output_dir,
        resolved_device=resolved_device,
        load_dtype=load_dtype,
        tokenizer_id=tokenizer_id,
        max_sequence_length=max_sequence_length,
        skip_conversion_validation=skip_conversion_validation,
    )
    return model, tokenizer, resolved_device, load_dtype, resolved_model_path, was_converted


def get_decoder_layers(model):
    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleDict):
        return IndexedLayerContainer(model.blocks)

    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("language_model", "model", "layers"),
    )

    for attr_chain in candidates:
        current = model
        for attr in attr_chain:
            current = getattr(current, attr, None)
            if current is None:
                break
        if isinstance(current, nn.ModuleList):
            return current

    raise AttributeError(f"Could not locate decoder layers for model class {model.__class__.__name__}")


def get_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)

    for attr in ("d_model", "hidden_size"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)

    raise AttributeError(f"Could not infer hidden size from model class {model.__class__.__name__}")


def is_moe_model(model) -> bool:
    config = getattr(model, "config", None)
    num_experts = getattr(config, "num_experts", None) if config is not None else None
    if isinstance(num_experts, int) and num_experts > 1:
        return True

    model_is_moe = getattr(model, "is_moe", None)
    if isinstance(model_is_moe, bool):
        return model_is_moe

    return any(hasattr(layer, "moe") or hasattr(layer, "feed_forward_moe") for layer in get_decoder_layers(model))


def _tokenize_text(tokenizer, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    token_ids = tokenized["input_ids"]
    if not token_ids:
        return []
    if isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return token_ids


def build_sample_batch(
    tokenizer,
    num_samples: int,
    seq_length: int,
    seed: int,
    text_file: str | None = None,
    prompts: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    texts: list[str]
    if text_file:
        raw_text = Path(text_file).expanduser().read_text(encoding="utf-8")
        texts = [chunk.strip() for chunk in raw_text.splitlines() if chunk.strip()]
        if not texts:
            texts = [raw_text.strip()]
    elif prompts:
        texts = [text.strip() for text in prompts if text.strip()]
    else:
        texts = DEFAULT_SAMPLE_TEXTS

    texts = [text for text in texts if text]
    if not texts:
        raise ValueError("No usable text was found for sampling.")

    eos_text = getattr(tokenizer, "eos_token", None) or "\n"
    corpus_text = f" {eos_text} ".join(texts)
    token_ids = _tokenize_text(tokenizer, corpus_text)

    if not token_ids:
        fallback_token = getattr(tokenizer, "eos_token_id", None)
        if fallback_token is None:
            raise ValueError("Tokenizer produced no tokens and has no eos_token_id fallback.")
        token_ids = [fallback_token]

    while len(token_ids) < seq_length + 1:
        token_ids = token_ids + token_ids

    rng = Random(seed)
    max_start = len(token_ids) - seq_length
    samples = []

    for _ in range(num_samples):
        start = rng.randint(0, max_start)
        sample = token_ids[start : start + seq_length]
        samples.append(torch.tensor(sample, dtype=torch.long))

    input_ids = torch.stack(samples)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _call_underlying_model(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool = False,
):
    backend = get_model_backend(model)
    if backend == "olmo_core":
        kwargs = {"input_ids": input_ids}
        if labels is not None:
            kwargs["labels"] = labels
        if use_cache:
            kwargs["use_cache"] = use_cache
        return model(**kwargs)

    kwargs = {
        "input_ids": input_ids,
        "use_cache": use_cache,
        "return_dict": True,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if labels is not None:
        kwargs["labels"] = labels
    return model(**kwargs)


def _capture_hidden_states(model):
    layers = get_decoder_layers(model)
    hidden_states: list[torch.Tensor | None] = [None] * (len(layers) + 1)
    hook_handles = []

    def pre_hook(module, module_input):
        if hidden_states[0] is not None:
            return
        value = module_input[0] if isinstance(module_input, tuple) else module_input
        hidden_states[0] = normalize_layer_output(value)

    if len(layers) > 0:
        hook_handles.append(layers[0].register_forward_pre_hook(pre_hook))

    for layer_idx, layer in enumerate(layers):
        def hook_fn(module, module_input, module_output, idx=layer_idx):
            hidden_states[idx + 1] = normalize_layer_output(module_output)

        hook_handles.append(layer.register_forward_hook(hook_fn))

    return hidden_states, hook_handles


def _package_model_outputs(raw_outputs, hidden_states: tuple[torch.Tensor | None, ...] | None = None):
    if isinstance(raw_outputs, torch.Tensor):
        logits = raw_outputs
        loss = None
    else:
        logits = getattr(raw_outputs, "logits", None)
        loss = getattr(raw_outputs, "loss", None)

    return SimpleNamespace(
        logits=logits,
        loss=loss,
        hidden_states=hidden_states,
        raw_outputs=raw_outputs,
    )


def model_forward(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    output_hidden_states: bool = False,
    use_cache: bool = False,
):
    hidden_states = None
    hook_handles = []
    captured_states = None

    if output_hidden_states:
        captured_states, hook_handles = _capture_hidden_states(model)

    try:
        raw_outputs = _call_underlying_model(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    if captured_states is not None:
        first_valid = next((state for state in captured_states if state is not None), None)
        if first_valid is not None:
            hidden_states = tuple(
                state if state is not None else torch.zeros_like(first_valid) for state in captured_states
            )
        else:
            hidden_states = tuple(captured_states)

    return _package_model_outputs(raw_outputs, hidden_states=hidden_states)


@contextmanager
def replace_decoder_layer(model, layer_idx: int, new_layer: nn.Module):
    layers = get_decoder_layers(model)
    original_layer = layers[layer_idx]
    layers[layer_idx] = new_layer
    try:
        yield original_layer
    finally:
        layers[layer_idx] = original_layer


def state_dict_clone(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def restore_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(state_dict, strict=True)


def layer_forward_returns_attentions(kwargs: dict) -> bool:
    return bool(kwargs.get("output_attentions", False))


def normalize_layer_output(output):
    if isinstance(output, tuple):
        return output[0] if output else None
    return output


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
