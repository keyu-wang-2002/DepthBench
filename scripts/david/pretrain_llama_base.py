#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import rich

DEPTHBENCH_ROOT = Path(__file__).resolve().parents[2]
OLMO_CORE_SRC = DEPTHBENCH_ROOT / "pretrain" / "OLMo-core" / "src"
for path in (DEPTHBENCH_ROOT, OLMO_CORE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_utils.model_config import load_llama_like_kwargs
from config_utils.tokenizer_config import build_tokenizer_config, copy_tokenizer_to_dir
from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
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
PROJECT_CODE_ROOT = "DepthBench"
TRAIN_DATA_GLOB = f"{PRETOKENIZED_DATA_ROOT}/train/*.npy"
EVAL_DATA_GLOB = f"{PRETOKENIZED_DATA_ROOT}/eval/*.npy"
TOKENIZER_PATH = f"{PROJECT_CODE_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
DEFAULT_MODEL_CONFIG_PATH = (
    f"{PROJECT_CODE_ROOT}/configs/llama_60M_backbone.json"
)
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


def build_config(args: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    save_folder = args.save_folder or f"workspace/{args.run_name}"
    work_dir = args.work_dir or str(Path(save_folder) / "dataset-cache")

    tokenizer_config = build_tokenizer_config(args.tokenizer_name_or_path)

    model_kwargs, raw_model_config = load_llama_like_kwargs(
        args.model_config,
        tokenizer_vocab_size=tokenizer_config.vocab_size,
    )

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


def maybe_copy_tokenizer(tokenizer_name_or_path: str, save_folder: str) -> None:
    destination_dir = Path(save_folder) / "tokenizer"
    copy_tokenizer_to_dir(tokenizer_name_or_path, destination_dir)


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

    if get_rank() == 0:
        maybe_copy_tokenizer(
            tokenizer_name_or_path=str(config.dataset.tokenizer.identifier),
            save_folder=trainer.save_folder,
        )

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
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
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
