#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig, FeedForwardType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    LayerStatsMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

PRETOKENIZED_DATA_ROOT = "/fast/wangk/data/fineweb-edu/pre-tokenize"
PROJECT_CODE_ROOT = "/home/wangk/DepthBench"
TRAIN_DATA_GLOB = f"{PRETOKENIZED_DATA_ROOT}/train/*.npy"
EVAL_DATA_GLOB = f"{PRETOKENIZED_DATA_ROOT}/eval/*.npy"
TOKENIZER_PATH = f"{PROJECT_CODE_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
DEFAULT_MODEL_CONFIG_PATH = (
    f"{PROJECT_CODE_ROOT}/configs/llama_60M_backbone.json"
)

HF_TO_LLAMA_LIKE_KEY_MAP = {
    "hidden_size": "d_model",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "rms_norm_eps": "layer_norm_eps",
    "torch_dtype": "dtype",
    "initializer_range": "init_std",
}

DIRECT_LLAMA_LIKE_KEYS = {
    "d_model",
    "vocab_size",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "qk_norm",
    "use_head_qk_norm",
    "layer_norm_eps",
    "rope_theta",
    "no_global_rope",
    "hidden_size_multiple_of",
    "hidden_size_multiplier",
    "fused_ops",
    "use_flash",
    "init_std",
    "embedding_init_std",
    "embed_scale",
}


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 42
    load_path: Optional[str] = None
    load_trainer_state: bool = False


def _load_llama_like_kwargs(config_path: str, default_vocab_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    activation_aliases = {
        "silu": ActivationFunction.silu,
        "swiglu": ActivationFunction.silu,
        "gelu_tanh": ActivationFunction.gelu_tanh,
        "gelu_pytorch_tanh": ActivationFunction.gelu_tanh,
    }

    def resolve_dtype(value: Any, default: DType = DType.bfloat16) -> DType:
        if value is None:
            return default
        if isinstance(value, DType):
            return value
        if isinstance(value, str):
            normalized = value.removeprefix("torch.")
            try:
                return DType(normalized)
            except ValueError as exc:
                raise ValueError(f"Unsupported dtype in model config: {value}") from exc
        raise TypeError(f"Unsupported dtype value in model config: {value!r}")

    def resolve_activation(value: Any) -> ActivationFunction:
        if value is None:
            return ActivationFunction.silu
        if isinstance(value, ActivationFunction):
            return value
        if isinstance(value, str) and value in activation_aliases:
            return activation_aliases[value]
        raise ValueError(f"Unsupported hidden_act in model config: {value}")

    def build_feed_forward_config(
        raw_config: dict[str, Any], dtype: DType
    ) -> Optional[FeedForwardConfig]:
        feed_forward = raw_config.get("feed_forward")
        if feed_forward is not None:
            if not isinstance(feed_forward, dict):
                raise TypeError("'feed_forward' in model config must be a JSON object")

            ff_kwargs = dict(feed_forward)
            ff_kwargs["dtype"] = resolve_dtype(ff_kwargs.get("dtype"), default=dtype)
            ff_kwargs["activation"] = resolve_activation(
                ff_kwargs.get("activation", raw_config.get("hidden_act"))
            )
            if "name" in ff_kwargs:
                ff_kwargs["name"] = FeedForwardType(ff_kwargs["name"])
            return FeedForwardConfig(**ff_kwargs)

        intermediate_size = raw_config.get("intermediate_size")
        if intermediate_size is None:
            return None

        return FeedForwardConfig(
            hidden_size=intermediate_size,
            bias=raw_config.get("mlp_bias", False),
            dtype=dtype,
            activation=resolve_activation(raw_config.get("hidden_act")),
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if raw_config.get("attention_bias") not in (None, False):
        raise ValueError(
            "This script only supports attention_bias=false because TransformerConfig.llama_like "
            "builds a bias-free attention module."
        )
    if raw_config.get("tie_word_embeddings") not in (None, False):
        raise ValueError(
            "This script does not support tie_word_embeddings=true with TransformerConfig.llama_like."
        )

    model_kwargs: dict[str, Any] = {}

    for key in DIRECT_LLAMA_LIKE_KEYS:
        value = raw_config.get(key)
        if value is not None:
            model_kwargs[key] = value

    for source_key, target_key in HF_TO_LLAMA_LIKE_KEY_MAP.items():
        if target_key not in model_kwargs and raw_config.get(source_key) is not None:
            model_kwargs[target_key] = raw_config[source_key]

    model_kwargs["dtype"] = resolve_dtype(model_kwargs.get("dtype"), default=DType.bfloat16)
    model_kwargs["vocab_size"] = model_kwargs.get("vocab_size", default_vocab_size)

    feed_forward = build_feed_forward_config(raw_config, model_kwargs["dtype"])
    if feed_forward is not None:
        model_kwargs["feed_forward"] = feed_forward

    required_keys = ("d_model", "n_layers", "n_heads", "vocab_size")
    missing_keys = [key for key in required_keys if model_kwargs.get(key) is None]
    if missing_keys:
        raise ValueError(
            f"Missing required model config fields for llama_like(): {', '.join(missing_keys)}"
        )

    return model_kwargs, raw_config


def build_config(args: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    save_folder = args.save_folder or f"workspace/{args.run_name}"
    work_dir = args.work_dir or str(Path(save_folder) / "dataset-cache")

    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    tokenizer_config.identifier = args.tokenizer_name_or_path

    model_kwargs, raw_model_config = _load_llama_like_kwargs(
        args.model_config,
        default_vocab_size=tokenizer_config.vocab_size,
    )

    if raw_model_config.get("vocab_size") is not None and raw_model_config["vocab_size"] != tokenizer_config.vocab_size:
        log.warning("Model vocab_size (%s) differs from tokenizer vocab_size (%s); using the value from the JSON config.", raw_model_config["vocab_size"], tokenizer_config.vocab_size)

    sequence_length = args.sequence_length or raw_model_config.get("max_sequence_length", 2048)
    model_config = TransformerConfig.llama_like(**model_kwargs)

    global_batch_size_tokens = args.global_train_batch_size * sequence_length
    rank_microbatch_size_tokens = args.device_train_microbatch_size * sequence_length

    dataset_config = NumpyFSLDatasetConfig.glob(
        args.train_data_glob,
        sequence_length=sequence_length,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size_tokens,
        seed=args.seed,
        num_workers=args.data_loader_num_workers,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size_tokens,
        max_sequence_length=sequence_length,
        optim=AdamWConfig(
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        ),
        scheduler=CosWithWarmup(
            warmup=args.warmup_steps,
            alpha_f=0.1,
            warmup_min_lr=0.0,
        ),
        max_grad_norm=1.0,
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        autocast_precision=DType.bfloat16,
    )

    eval_duration = (
        Duration.steps(args.eval_max_batches)
        if args.eval_max_batches > 0
        else Duration.epochs(1)
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            max_duration=Duration.steps(args.max_steps),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            bookkeeping_soft_timeout=120,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "layer_stats",
            LayerStatsMonitorCallback(
                enabled=args.enable_layer_stats,
                interval=args.layer_stats_interval,
            ),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=args.save_interval,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=args.run_name,
                project=args.wandb_project,
                entity=args.wandb_entity,
                cancel_check_interval=10,
                enabled=bool(args.wandb_project),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.glob(
                    args.eval_data_glob,
                    metadata=[{"label": "fineweb-edu-eval"}],
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_interval=args.eval_interval,
                eval_duration=eval_duration,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=args.seed,
        load_path=args.load_path,
        load_trainer_state=args.load_trainer_state,
    ).merge(overrides)


def train(config: ExperimentConfig) -> None:
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info("Loading checkpoint from %s", config.load_path)
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)

    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a llama-like baseline on pre-tokenized numpy data with a JSON model config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_name", nargs="?", default="pretrain-llama-base")
    parser.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--train-data-glob", type=str, default=TRAIN_DATA_GLOB)
    parser.add_argument("--eval-data-glob", type=str, default=EVAL_DATA_GLOB)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--global-train-batch-size", type=int, default=512)
    parser.add_argument("--device-train-microbatch-size", type=int, default=16)
    parser.add_argument("--data-loader-num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=160)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-max-batches", type=int, default=-1)
    parser.add_argument("--save-interval", type=int, default=400)
    parser.add_argument("--wandb-project", type=str, default="residual-bench")
    parser.add_argument("--wandb-entity", type=str, default="wang-keyu-2002-max-planck-society")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--load-trainer-state", action="store_true")
    parser.add_argument("--enable-layer-stats", action="store_true")
    parser.add_argument("--layer-stats-interval", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args()
    config = build_config(args, overrides)

    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
