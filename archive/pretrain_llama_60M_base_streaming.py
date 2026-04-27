"""
This file uses streaming to read parquet files and perform tokenization.
This method has been deprecated in this project, 
and the file is retained only as a draft.

In this project, pre-tokenization is used 
along with the built-in data loading method of OLMo-core
refer ./pretrain_llama_60M_base.py
and https://olmo-core.readthedocs.io/en/stable/guides/data_loading.html
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import StreamingParquetDataLoaderConfig, TokenizerConfig
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
    ParquetLMEvalCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

RAW_DATA_ROOT = ""
PROJECT_CODE_ROOT = ""
TRAIN_PARQUET_GLOB = f"{RAW_DATA_ROOT}/100BT/*.parquet"
EVAL_PARQUET_PATH = f"{RAW_DATA_ROOT}/eval/eval_013_00008.parquet"
TOKENIZER_PATH = f"{PROJECT_CODE_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    data_loader: StreamingParquetDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 6198
    load_path: Optional[str] = None


def build_config(args: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    save_folder = args.save_folder or f"workspace/{args.run_name}"
    work_dir = args.work_dir or str(Path(save_folder) / "dataset-cache")

    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    tokenizer_config.identifier = args.tokenizer_name_or_path

    model_config = TransformerConfig.llama_60M_backbone(
        vocab_size=tokenizer_config.vocab_size,
        dtype=DType.bfloat16,
    )

    global_batch_size_tokens = args.global_train_batch_size * args.sequence_length
    rank_microbatch_size_tokens = args.device_train_microbatch_size * args.sequence_length

    data_loader_config = StreamingParquetDataLoaderConfig(
        train_parquet_glob=args.train_parquet_glob,
        eval_parquet_path=args.eval_parquet_path,
        text_field=args.text_field,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        tokenizer_config=tokenizer_config,
        sequence_length=args.sequence_length,
        global_batch_size=global_batch_size_tokens,
        work_dir=work_dir,
        seed=args.seed,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size_tokens,
        max_sequence_length=args.sequence_length,
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
            ParquetLMEvalCallback(
                eval_parquet_path=args.eval_parquet_path,
                text_field=args.text_field,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                tokenizer_config=tokenizer_config,
                sequence_length=args.sequence_length,
                eval_interval=args.eval_interval,
                eval_max_batches=args.eval_max_batches,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=args.seed,
    ).merge(overrides)


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    data_loader = config.data_loader.build(dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info("Loading checkpoint from %s", config.load_path)
        trainer.load_checkpoint(config.load_path)

    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the llama-60M baseline on parquet data with DDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_name", nargs="?", default="pretrain-llama-60M-new")
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--train-parquet-glob", type=str, default=TRAIN_PARQUET_GLOB)
    parser.add_argument("--eval-parquet-path", type=str, default=EVAL_PARQUET_PATH)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--global-train-batch-size", type=int, default=512)
    parser.add_argument("--device-train-microbatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=320)
    parser.add_argument("--eval-interval", type=int, default=400)
    parser.add_argument("--eval-max-batches", type=int, default=-1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
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
