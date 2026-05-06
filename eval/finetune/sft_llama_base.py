#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import rich
from transformers import AutoTokenizer, PreTrainedTokenizerFast

DEPTHBENCH_ROOT = Path(__file__).resolve().parents[2]
OLMO_CORE_SRC = DEPTHBENCH_ROOT / "pretrain" / "OLMo-core" / "src"
if str(DEPTHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHBENCH_ROOT))
if str(OLMO_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(OLMO_CORE_SRC))

from config_utils.model_config import load_llama_like_kwargs
from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

DEFAULT_TOKENIZER_PATH = (
    OLMO_CORE_SRC
    / "olmo_core"
    / "data"
    / "tokenizers"
    / "allenai_gpt-neox-olmo-dolma-v1_5.json"
)
DEFAULT_MODEL_CONFIG_PATH = DEPTHBENCH_ROOT / "configs" / "llama_60M_backbone.json"
SPECIAL_TOKEN_CANDIDATES = {
    "eos_token": ("<|endoftext|>", "</s>"),
    "pad_token": ("<|padding|>", "<pad>"),
    "bos_token": ("<s>", "<bos>"),
}


def _token_exists(tokenizer, token: str) -> bool:
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id is not None and token_id != unk_token_id


def _ensure_special_tokens(tokenizer):
    for token_attr, candidates in SPECIAL_TOKEN_CANDIDATES.items():
        if getattr(tokenizer, f"{token_attr}_id", None) is not None:
            continue
        for candidate in candidates:
            if _token_exists(tokenizer, candidate):
                setattr(tokenizer, token_attr, candidate)
                break
    return tokenizer


def load_hf_tokenizer(tokenizer_name_or_path: str):
    tokenizer_path = Path(tokenizer_name_or_path).expanduser()
    tokenizer_json_path = tokenizer_path / "tokenizer.json"

    if tokenizer_path.is_file() and tokenizer_path.suffix == ".json":
        return _ensure_special_tokens(
            PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
        )
    if tokenizer_path.is_dir() and tokenizer_json_path.is_file():
        return _ensure_special_tokens(
            PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json_path))
        )
    return _ensure_special_tokens(
        AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            local_files_only=tokenizer_path.exists(),
        )
    )


def choose_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
        return int(tokenizer.vocab_size)
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    return int(len(tokenizer))


def resolve_tokenizer_field(
    *,
    tokenizer_value: Optional[int],
    override_value: Optional[int],
    field_name: str,
    required: bool,
) -> Optional[int]:
    if tokenizer_value is not None:
        tokenizer_value = int(tokenizer_value)
    if override_value is not None:
        override_value = int(override_value)

    if tokenizer_value is not None and override_value is not None and tokenizer_value != override_value:
        raise ValueError(
            f"Tokenizer {field_name} ({tokenizer_value}) does not match CLI override ({override_value}). "
            f"Update the tokenizer or remove --{field_name.replace('_', '-')}."
        )

    if tokenizer_value is not None:
        return tokenizer_value
    if override_value is not None:
        return override_value
    if required:
        raise ValueError(
            f"Tokenizer is missing {field_name}. Provide --{field_name.replace('_', '-')} explicitly."
        )
    return None


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyPackedFSLDatasetConfig | NumpyPaddedFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 42
    pretrain_checkpoint: Optional[str] = None


def build_tokenizer_config(args: argparse.Namespace) -> TokenizerConfig:
    tokenizer = load_hf_tokenizer(args.tokenizer_name_or_path)
    vocab_size = choose_vocab_size(tokenizer)
    if args.tokenizer_vocab_size is not None and int(args.tokenizer_vocab_size) != vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({vocab_size}) does not match CLI override ({args.tokenizer_vocab_size}). "
            "Update the tokenizer or remove --tokenizer-vocab-size."
        )

    tokenizer_config = TokenizerConfig(
        vocab_size=vocab_size,
        eos_token_id=cast(
            int,
            resolve_tokenizer_field(
                tokenizer_value=getattr(tokenizer, "eos_token_id", None),
                override_value=args.eos_token_id,
                field_name="eos_token_id",
                required=True,
            ),
        ),
        pad_token_id=cast(
            int,
            resolve_tokenizer_field(
                tokenizer_value=getattr(tokenizer, "pad_token_id", None),
                override_value=args.pad_token_id,
                field_name="pad_token_id",
                required=True,
            ),
        ),
        bos_token_id=resolve_tokenizer_field(
            tokenizer_value=getattr(tokenizer, "bos_token_id", None),
            override_value=args.bos_token_id,
            field_name="bos_token_id",
            required=False,
        ),
        identifier=args.tokenizer_name_or_path,
    )
    return tokenizer_config


def build_dp_config(world_size: int, local_world_size: int) -> TransformerDataParallelConfig:
    if world_size <= 1 or args_disable_hsdp():
        return TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        )

    shard_degree = max(1, min(local_world_size, world_size))
    return TransformerDataParallelConfig(
        name=DataParallelType.hsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        shard_degree=shard_degree,
    )


def args_disable_hsdp() -> bool:
    return os.environ.get("DEPTHBENCH_DISABLE_HSDP", "").lower() in {"1", "true", "yes"}


def build_config(args: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    save_folder = args.save_folder or f"workspace/{args.run_name}"
    work_dir = args.work_dir or str(Path(save_folder) / "dataset-cache")

    tokenizer_config = build_tokenizer_config(args)
    model_kwargs, raw_model_config = load_llama_like_kwargs(
        args.model_config,
        tokenizer_vocab_size=tokenizer_config.vocab_size,
    )

    sequence_length = args.sequence_length or raw_model_config.get("max_sequence_length", 512)
    model_config = TransformerConfig.llama_like(**model_kwargs)

    dataset_paths = [str(Path(args.dataset_dir) / "token_ids_part_*.npy")]
    label_mask_paths = [str(Path(args.dataset_dir) / "labels_mask_part_*.npy")]
    if args.dataset_layout == "packed":
        dataset_config = NumpyPackedFSLDatasetConfig(
            tokenizer=tokenizer_config,
            work_dir=work_dir,
            paths=dataset_paths,
            expand_glob=True,
            label_mask_paths=label_mask_paths,
            generate_doc_lengths=True,
            long_doc_strategy=LongDocStrategy.truncate,
            sequence_length=sequence_length,
        )
    else:
        dataset_config = NumpyPaddedFSLDatasetConfig(
            tokenizer=tokenizer_config,
            work_dir=work_dir,
            paths=dataset_paths,
            expand_glob=True,
            label_mask_paths=label_mask_paths,
            sequence_length=sequence_length,
        )

    global_batch_size_tokens = args.global_train_batch_size * sequence_length
    rank_microbatch_size_tokens = args.device_train_microbatch_size * sequence_length

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size_tokens,
        seed=args.seed,
        num_workers=args.data_loader_num_workers,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    dp_config = build_dp_config(world_size=world_size, local_world_size=local_world_size)

    ac_config = None
    if not args.disable_activation_checkpointing:
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size_tokens,
        max_sequence_length=sequence_length,
        z_loss_multiplier=None,
        compile_model=not args.disable_compile,
        optim=SkipStepAdamWConfig(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            compile=False,
        ),
        scheduler=LinearWithWarmup(
            warmup_fraction=args.warmup_fraction,
            alpha_f=0.1,
            warmup_min_lr=0.0,
        ),
        max_grad_norm=args.max_grad_norm,
        dp_config=dp_config,
        ac_config=ac_config,
        autocast_precision=DType.bfloat16,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            max_duration=Duration.epochs(args.epochs),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            bookkeeping_soft_timeout=120,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=args.save_interval,
                ephemeral_save_interval=args.ephemeral_save_interval,
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
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=args.seed,
        pretrain_checkpoint=args.pretrain_checkpoint,
    ).merge(overrides)


def maybe_copy_tokenizer(dataset_dir: str, save_folder: str) -> None:
    source_dir = Path(dataset_dir) / "tokenizer"
    destination_dir = Path(save_folder) / "tokenizer"
    if not source_dir.is_dir() or destination_dir.exists():
        return
    tokenizer_json_path = source_dir / "tokenizer.json"
    if tokenizer_json_path.is_file() and not (source_dir / "tokenizer_config.json").exists():
        load_hf_tokenizer(str(source_dir)).save_pretrained(str(destination_dir))
    else:
        shutil.copytree(source_dir, destination_dir)


def train(config: ExperimentConfig, dataset_dir: str) -> None:
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
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict

    if get_rank() == 0:
        maybe_copy_tokenizer(dataset_dir=dataset_dir, save_folder=trainer.save_folder)

    if not trainer.maybe_load_checkpoint():
        if not config.pretrain_checkpoint:
            raise ValueError("No saved SFT checkpoint found and --pretrain-checkpoint was not set.")
        log.info("Loading pretrain checkpoint from %s", config.pretrain_checkpoint)
        trainer.load_checkpoint(config.pretrain_checkpoint, load_trainer_state=False, load_optim_state=False)
    else:
        log.info("Resumed from an existing SFT checkpoint under %s", trainer.save_folder)

    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full SFT from a pre-train checkpoint using OLMo-core packed SFT data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", type=str, default="llama-commonsense170k-sft")
    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--pretrain-checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=str(DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--tokenizer-vocab-size", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument("--bos-token-id", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--global-train-batch-size", type=int, default=128)
    parser.add_argument("--device-train-microbatch-size", type=int, default=16)
    parser.add_argument("--data-loader-num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=None, help="Permanent checkpoint interval in steps. Leave unset to disable intermediate saves.")
    parser.add_argument("--ephemeral-save-interval", type=int, default=None, help="Ephemeral checkpoint interval in steps. Leave unset to disable temporary saves.")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--dataset-layout", type=str, choices=("padded", "packed"), default="padded", help="Use 'packed' only with an attention backend that supports intra-document masking.")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-activation-checkpointing", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args()
    config = build_config(args, overrides)

    prepare_training_environment()
    try:
        train(config, dataset_dir=args.dataset_dir)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
