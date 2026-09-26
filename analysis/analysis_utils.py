from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLMO_CORE_SRC = Path(
    os.environ.get("DEPTHBENCH_OLMO_CORE_SRC", str(PROJECT_ROOT / "pretrain" / "OLMo-core" / "src"))
)
if str(OLMO_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(OLMO_CORE_SRC))

from olmo_core.data.parquet import load_parquet_tokenizer
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.nn.transformer.config import TransformerConfig


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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_experiment_config(config: dict) -> dict:
    """Make configs written by older DepthBench revisions buildable with the vendored OLMo-core."""
    tokenizer_config = config.get("dataset", {}).get("tokenizer", {})
    tokenizer_identifier = tokenizer_config.get("identifier")
    if isinstance(tokenizer_identifier, str) and tokenizer_identifier.endswith(".json"):
        # Tokenizer files may be recorded as absolute paths from another machine or as
        # paths relative to the repository root.
        tokenizer_path = Path(tokenizer_identifier)
        candidates = [
            tokenizer_path,
            PROJECT_ROOT / tokenizer_path,
            OLMO_CORE_SRC / "olmo_core" / "data" / "tokenizers" / tokenizer_path.name,
        ]
        resolved = next((path for path in candidates if path.exists()), None)
        if resolved is not None:
            tokenizer_config["identifier"] = str(resolved.resolve())

    block_config = config.get("model", {}).get("block", {})
    if block_config.get("name") not in {"moda", "post_norm_moda"}:
        # Older Pre-LN checkpoints may carry the (unused) MoDA-only key.
        block_config.pop("moda_skip_ffn_kv", None)
    sequence_mixer = block_config.get("sequence_mixer", {})
    if sequence_mixer.get("backend") == "flash_2":
        sequence_mixer["backend"] = "torch"
    # AttnRes checkpoints used to be stored as `default` (or `attn_res`) + attnres_block_size.
    if block_config.get("name") == "attn_res":
        block_config["name"] = "attnres"
    if block_config.get("name") == "default" and block_config.get("attnres_block_size") is not None:
        block_config["name"] = "attnres"
    return config


def looks_like_olmo_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists():
        return False

    config = _load_json(config_path)
    return isinstance(config, dict) and "model" in config and "dataset" in config


def resolve_checkpoint_dir(model_path: str) -> Path:
    """Accept a checkpoint dir, or a run dir whose latest `step*` subdirectory is used."""
    checkpoint_dir = Path(model_path).expanduser().resolve()
    if looks_like_olmo_checkpoint(checkpoint_dir):
        return checkpoint_dir

    step_dirs = []
    for child in checkpoint_dir.glob("step*"):
        try:
            step = int(child.name.removeprefix("step"))
        except ValueError:
            continue
        if child.is_dir() and looks_like_olmo_checkpoint(child):
            step_dirs.append((step, child))
    if not step_dirs:
        raise ValueError(
            f"Unsupported model path: {checkpoint_dir}. Expected an OLMo-core checkpoint directory "
            "containing config.json with 'model' and 'dataset' entries, or a run directory with step* checkpoints."
        )
    return max(step_dirs)[1]


def load_experiment_config(model_path: str) -> tuple[Path, dict]:
    checkpoint_dir = resolve_checkpoint_dir(model_path)
    return checkpoint_dir, normalize_experiment_config(_load_json(checkpoint_dir / "config.json"))


def _torch_fused_attnres(
    query: torch.Tensor,
    residuals,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
):
    """PyTorch implementation of `olmo_core.kernels.attnres.fused_attnres`."""
    if len(residuals) == 0:
        raise ValueError("residuals must contain at least one source")
    query_vector = query.reshape(-1).float()
    sources = torch.stack([residual.float() for residual in residuals], dim=0)
    normalized = sources * torch.rsqrt(sources.square().mean(dim=-1, keepdim=True) + rms_eps)
    scores = (normalized * rms_weight.float() * query_vector).sum(dim=-1) * scale
    probs = torch.softmax(scores, dim=0)
    mixed = (probs.unsqueeze(-1) * sources).sum(dim=0)
    if output_rms_weight is not None:
        mixed = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + rms_eps)
        mixed = mixed * output_rms_weight.float()
    mixed = mixed.to(residuals[0].dtype)
    if return_weights:
        return mixed, probs
    return mixed


def _install_attnres_fallback(force: bool) -> None:
    try:
        import olmo_core.kernels.attnres as attnres_kernel
    except Exception:
        # The Triton kernel lives in `fla`; analysis only needs a functional implementation.
        attnres_kernel = types.ModuleType("olmo_core.kernels.attnres")
        sys.modules["olmo_core.kernels.attnres"] = attnres_kernel
        force = True
    if force:
        attnres_kernel.fused_attnres = _torch_fused_attnres


def _install_liger_mhc_fallback() -> None:
    """Replace the Liger mHC kernels with the equivalent PyTorch forward (no Triton needed)."""
    from olmo_core.nn import hyper_connections as hc

    liger_cls = getattr(hc, "LigerHyperConnection", None)
    if liger_cls is None or getattr(liger_cls, "_depthbench_torch_forward", False):
        return

    def forward(self, residuals: torch.Tensor, *args, **kwargs):
        from torch.utils._pytree import tree_flatten, tree_unflatten

        streams = hc._reshape_to_streams(residuals, self.num_residual_streams).float()
        normed = hc._rms_norm(streams.reshape(*streams.shape[:-2], -1), eps=self.rms_eps)
        mix = torch.matmul(normed, self.phi.float()) + self.b.float()
        n = self.num_residual_streams
        h_pre = torch.sigmoid(self.alpha_pre.float() * mix[..., :n] + self.pre_eps)
        h_post = self.post_mult * torch.sigmoid(self.alpha_post.float() * mix[..., n : 2 * n])
        h_res_logits = mix[..., 2 * n : 2 * n + n * n].reshape(*mix.shape[:-1], n, n)
        h_res = hc.sinkhorn_log(self.alpha_res.float() * h_res_logits, num_iters=self.tmax, tau=1.0)

        branch_input = torch.einsum("...s,...sd->...d", h_pre, streams).to(residuals.dtype)
        branch_input = branch_input.to(self._branch_param_dtype(branch_input.dtype))
        branch_output = self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)
        mixed = torch.einsum("...ts,...sd->...td", h_res, streams)
        output = mixed + branch_output.float().unsqueeze(-2) * h_post.unsqueeze(-1)
        output = self.dropout(hc._flatten_from_streams(output).to(dtype=residuals.dtype))
        return tree_unflatten((output, *rest), tree_spec)

    liger_cls.forward = forward
    liger_cls._depthbench_torch_forward = True


def install_runtime_fallbacks(device: str) -> None:
    """Use PyTorch implementations of Triton-only kernels where needed.

    On CPU both AttnRes and Liger mHC fall back to PyTorch. On GPU the fused kernels are
    used unless DEPTHBENCH_USE_LIGER_MHC_FALLBACK=1 requests the PyTorch mHC forward.
    """
    on_cpu = get_device_type(device) == "cpu"
    _install_attnres_fallback(force=on_cpu)
    if on_cpu or os.environ.get("DEPTHBENCH_USE_LIGER_MHC_FALLBACK") == "1":
        _install_liger_mhc_fallback()


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype: str,
    tokenizer_id: str | None,
    max_sequence_length: int | None,
):
    resolved_device = resolve_device(device)
    install_runtime_fallbacks(resolved_device)
    load_dtype = resolve_load_dtype(dtype, resolved_device)
    checkpoint_dir, experiment_config = load_experiment_config(model_path)

    model_config = TransformerConfig.from_dict(experiment_config["model"])
    tokenizer_config = TokenizerConfig.from_dict(experiment_config["dataset"]["tokenizer"])
    tokenizer_name_or_path = tokenizer_id or tokenizer_config.identifier
    if tokenizer_name_or_path is None:
        raise ValueError(
            "Tokenizer identifier is missing from the checkpoint config and no --tokenizer-id was provided."
        )

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
    return model, tokenizer, resolved_device, load_dtype, str(checkpoint_dir)


def get_decoder_layers(model):
    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleDict):
        return IndexedLayerContainer(model.blocks)
    raise AttributeError(
        f"Could not locate OLMo-core decoder layers for model class {model.__class__.__name__}"
    )


def get_hidden_size(model) -> int:
    for attr in ("d_model", "hidden_size"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)

    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("d_model", "hidden_size", "n_embd"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)

    raise AttributeError(f"Could not infer hidden size from model class {model.__class__.__name__}")


def is_moe_model(model) -> bool:
    return any(
        hasattr(layer, "moe") or hasattr(layer, "feed_forward_moe")
        for layer in get_decoder_layers(model)
    )


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


def sample_token_windows(paths: Sequence[Path], num_samples: int, seq_length: int, seed: int) -> torch.Tensor:
    """Sample random contiguous windows from pre-tokenized `.npy` shards (raw uint16 memmaps accepted)."""
    arrays = []
    for path in paths:
        try:
            arrays.append(np.load(path, mmap_mode="r").reshape(-1))
        except ValueError:
            arrays.append(np.memmap(path, mode="r", dtype=np.uint16).reshape(-1))

    usable = np.asarray([max(0, len(arr) - seq_length) for arr in arrays], dtype=np.float64)
    if usable.sum() <= 0:
        raise ValueError(f"Token data files are too short for seq_length={seq_length}")

    rng = np.random.default_rng(seed)
    file_ids = rng.choice(len(arrays), size=num_samples, p=usable / usable.sum())
    samples = []
    for file_id in file_ids:
        arr = arrays[int(file_id)]
        start = int(rng.integers(0, len(arr) - seq_length))
        samples.append(np.asarray(arr[start : start + seq_length], dtype=np.int64))
    return torch.from_numpy(np.stack(samples))


def add_model_args(parser: argparse.ArgumentParser, *, num_samples: int, seq_length: int) -> None:
    parser.add_argument("--model_path", type=str, required=True, help="OLMo-core checkpoint or run dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=num_samples, help="Number of token windows")
    parser.add_argument("--seq_length", type=int, default=seq_length, help="Token window length")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Sequences per forward pass")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--token-data-glob", type=str, default=None, help="Glob of pre-tokenized .npy shards")
    parser.add_argument("--text-file", type=str, default=None, help="UTF-8 text file used when no token data is given")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text; can be repeated")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--tokenizer-id", type=str, default=None, help="Optional tokenizer override")
    parser.add_argument("--max-sequence-length", type=int, default=None, help="Tokenizer model_max_length override")


def load_eval_input_ids(args: argparse.Namespace, tokenizer) -> tuple[torch.Tensor, dict]:
    """Build the evaluation token batch from `--token-data-glob`, `--text-file` or `--prompt`."""
    if args.token_data_glob:
        paths = [Path(path) for path in sorted(glob.glob(args.token_data_glob))]
        if not paths:
            raise ValueError(f"No token data files matched --token-data-glob={args.token_data_glob}")
        input_ids = sample_token_windows(paths, args.num_samples, args.seq_length, args.seed)
        source = {"sample_source": "token-data-glob", "token_data_glob": args.token_data_glob}
    else:
        batch = build_sample_batch(
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            text_file=args.text_file,
            prompts=args.prompt,
        )
        input_ids = batch["input_ids"]
        source = {"sample_source": "text-or-prompts", "text_file": args.text_file}
    source.update(num_samples=args.num_samples, seq_length=args.seq_length, seed=args.seed)
    return input_ids, source


def iter_micro_batches(input_ids: torch.Tensor, micro_batch_size: int, device: str) -> Iterator[torch.Tensor]:
    for start in range(0, input_ids.shape[0], micro_batch_size):
        yield input_ids[start : start + micro_batch_size].to(device)


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
    del attention_mask

    hidden_states = None
    hook_handles = []
    captured_states = None

    if output_hidden_states:
        captured_states, hook_handles = _capture_hidden_states(model)

    try:
        kwargs = {"input_ids": input_ids}
        if labels is not None:
            kwargs["labels"] = labels
        if use_cache:
            kwargs["use_cache"] = use_cache
        raw_outputs = model(**kwargs)
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
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
