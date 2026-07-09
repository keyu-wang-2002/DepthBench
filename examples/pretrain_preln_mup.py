#!/usr/bin/env python3
"""Train Pre-LN depth-muP and CompleteP variants."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from types import MethodType
from typing import List, Optional, Tuple, cast

import rich
import torch

import pretrain_llama_base as base
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.lm_head import LMHead
from olmo_core.nn.transformer import Transformer, TransformerBlockType
from olmo_core.optim import OptimGroupOverride
from olmo_core.train.callbacks import ConfigSaverCallback
from olmo_core.utils import seed_all

PARAMETERIZATION_ALIASES = {
    "depth_mup": "depth_mup",
    "depth-mup": "depth_mup",
    "depth-muP": "depth_mup",
    "completep": "completep",
    "completeP": "completep",
}

PARAMETERIZATION_TO_ALPHA = {
    "depth_mup": 0.5,
    "completep": 1.0,
}

SCALING_AXIS_ALIASES = {
    "depth_only": "depth_only",
    "depth-only": "depth_only",
    "depth_width": "depth_width",
    "depth-width": "depth_width",
}


@dataclass(frozen=True)
class DepthWidthScalingSpec:
    width_multiplier: float
    lm_head_forward_scale: float
    lm_head_init_rescale: float


def add_preln_mup_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--parameterization",
        required=True,
        help="Choose depth_mup/depth-muP or completep/completeP.",
    )
    parser.add_argument(
        "--scaling-axis",
        required=True,
        help="Choose depth_only/depth-only or depth_width/depth-width scaling.",
    )
    parser.add_argument(
        "--base-depth",
        type=int,
        default=24,
        help="Reference depth whose depth multiplier is one.",
    )
    parser.add_argument(
        "--base-width",
        type=int,
        default=1024,
        help="Reference width whose width multiplier is one; used for depth_width only.",
    )
    return parser


def _resolve_parameterization(value: str) -> tuple[TransformerBlockType, float]:
    parameterization = PARAMETERIZATION_ALIASES.get(value)
    if parameterization is None:
        raise ValueError(f"unsupported parameterization: {value}")
    block_type = (
        TransformerBlockType.depth_mup
        if parameterization == "depth_mup"
        else TransformerBlockType.completep
    )
    return block_type, PARAMETERIZATION_TO_ALPHA[parameterization]


def _resolve_scaling_axis(value: str) -> str:
    scaling_axis = SCALING_AXIS_ALIASES.get(value)
    if scaling_axis is None:
        raise ValueError(f"unsupported scaling axis: {value}")
    return scaling_axis


def _validate_base_depth(args: argparse.Namespace) -> None:
    if args.base_depth <= 0:
        raise ValueError(f"--base-depth must be positive, got {args.base_depth}")


def _set_depth_block_type(
    config: base.ExperimentConfig,
    args: argparse.Namespace,
    block_type: TransformerBlockType,
) -> None:
    _validate_base_depth(args)
    config.model.block.name = block_type
    config.model.block.residual_scaling_base_depth = args.base_depth


def apply_depth_only_scaling(
    config: base.ExperimentConfig,
    args: argparse.Namespace,
    *,
    block_type: TransformerBlockType,
    alpha: float,
) -> base.ExperimentConfig:
    """Apply depth-only optimizer rules with m_L = L / L_base."""

    _set_depth_block_type(config, args, block_type)

    depth_multiplier = config.model.n_layers / args.base_depth
    block_lr_scale = depth_multiplier ** (alpha - 1.0)
    block_eps_scale = depth_multiplier ** (-alpha)

    if not (
        math.isclose(block_lr_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(block_eps_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        config.train_module.optim.group_overrides = [
            OptimGroupOverride(
                params=["blocks.*"],
                opts={
                    "lr": args.learning_rate * block_lr_scale,
                    "eps": config.train_module.optim.eps * block_eps_scale,
                },
            )
        ]

    return config


def _block_matrix_patterns(config: base.ExperimentConfig) -> List[str]:
    patterns = [
        "blocks.*.attention.w_q.weight",
        "blocks.*.attention.w_k.weight",
        "blocks.*.attention.w_v.weight",
        "blocks.*.attention.w_out.weight",
        "blocks.*.feed_forward.w1.weight",
        "blocks.*.feed_forward.w2.weight",
        "blocks.*.feed_forward.w3.weight",
    ]
    if getattr(config.model.block.sequence_mixer, "gate", None) is not None:
        patterns.append("blocks.*.attention.w_g.weight")
    if getattr(config.model.block, "attnres_block_size", None) is not None:
        patterns.extend(["blocks.*.attn_res_proj.weight", "blocks.*.mlp_res_proj.weight"])
    return patterns


def _block_norm_patterns(config: base.ExperimentConfig) -> List[str]:
    patterns = ["blocks.*.attention_norm.weight", "blocks.*.feed_forward_norm.weight"]
    if getattr(config.model.block.sequence_mixer, "qk_norm", None) is not None:
        patterns.extend(["blocks.*.attention.q_norm.weight", "blocks.*.attention.k_norm.weight"])
    if getattr(config.model.block, "attnres_block_size", None) is not None:
        patterns.extend(["blocks.*.attn_res_norm.weight", "blocks.*.mlp_res_norm.weight"])
    return patterns


def _lm_head_patterns(config: base.ExperimentConfig) -> List[str]:
    patterns = ["lm_head.w_out.weight"]
    if getattr(config.model.lm_head, "layer_norm", None) is not None:
        patterns.append("lm_head.norm.weight")
    return patterns


def _add_override_if_needed(
    group_overrides: List[OptimGroupOverride],
    *,
    params: List[str],
    base_lr: float,
    base_eps: float,
    base_weight_decay: float,
    lr: Optional[float] = None,
    eps: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> None:
    opts = {}
    if lr is not None and not math.isclose(lr, base_lr, rel_tol=0.0, abs_tol=1e-12):
        opts["lr"] = lr
    if eps is not None and not math.isclose(eps, base_eps, rel_tol=0.0, abs_tol=1e-20):
        opts["eps"] = eps
    if weight_decay is not None and not math.isclose(
        weight_decay, base_weight_decay, rel_tol=0.0, abs_tol=1e-12
    ):
        opts["weight_decay"] = weight_decay
    if opts:
        group_overrides.append(OptimGroupOverride(params=params, opts=opts))


def apply_depth_width_scaling(
    config: base.ExperimentConfig,
    args: argparse.Namespace,
    *,
    block_type: TransformerBlockType,
    alpha: float,
) -> tuple[base.ExperimentConfig, DepthWidthScalingSpec]:
    """Apply depth-width optimizer/init rules with m_L = L / L_base and m_N = d / d_base."""

    _set_depth_block_type(config, args, block_type)
    if args.base_width <= 0:
        raise ValueError(f"--base-width must be positive, got {args.base_width}")

    depth_multiplier = config.model.n_layers / args.base_depth
    width_multiplier = config.model.d_model / args.base_width

    pre_ln_lr_scale = depth_multiplier ** (alpha - 1.0)
    hidden_lr_scale = (width_multiplier ** -1.0) * pre_ln_lr_scale
    hidden_wd_scale = width_multiplier
    residual_block_eps_scale = (width_multiplier ** -1.0) * depth_multiplier ** (-alpha)
    emb_unemb_eps_scale = width_multiplier ** -1.0
    hidden_init_scale = width_multiplier ** (-0.5)

    base_init_std = config.model.init_std
    config.model.embedding_init_std = (
        config.model.embedding_init_std
        if config.model.embedding_init_std is not None
        else base_init_std
    )
    config.model.init_std = base_init_std * hidden_init_scale

    base_lr = args.learning_rate
    base_eps = config.train_module.optim.eps
    base_weight_decay = config.train_module.optim.weight_decay

    group_overrides: List[OptimGroupOverride] = []
    _add_override_if_needed(
        group_overrides,
        params=_block_matrix_patterns(config),
        base_lr=base_lr,
        base_eps=base_eps,
        base_weight_decay=base_weight_decay,
        lr=base_lr * hidden_lr_scale,
        eps=base_eps * residual_block_eps_scale,
        weight_decay=base_weight_decay * hidden_wd_scale,
    )
    _add_override_if_needed(
        group_overrides,
        params=_block_norm_patterns(config),
        base_lr=base_lr,
        base_eps=base_eps,
        base_weight_decay=base_weight_decay,
        lr=base_lr * pre_ln_lr_scale,
        eps=base_eps * residual_block_eps_scale,
    )
    _add_override_if_needed(
        group_overrides,
        params=["embeddings.weight"],
        base_lr=base_lr,
        base_eps=base_eps,
        base_weight_decay=base_weight_decay,
        eps=base_eps * emb_unemb_eps_scale,
    )
    _add_override_if_needed(
        group_overrides,
        params=_lm_head_patterns(config),
        base_lr=base_lr,
        base_eps=base_eps,
        base_weight_decay=base_weight_decay,
        eps=base_eps * emb_unemb_eps_scale,
    )

    config.train_module.optim.group_overrides = group_overrides or None

    return config, DepthWidthScalingSpec(
        width_multiplier=width_multiplier,
        lm_head_forward_scale=width_multiplier ** -1.0,
        lm_head_init_rescale=width_multiplier ** 0.5,
    )


def _patch_lm_head_forward(lm_head: LMHead, *, scale: float) -> None:
    if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return

    original_forward = lm_head.w_out.forward

    def scaled_forward(self, x):
        return original_forward(x) * scale

    lm_head.w_out.forward = MethodType(scaled_forward, lm_head.w_out)


def _patch_model_init(model: Transformer, *, lm_head_init_rescale: float) -> None:
    if math.isclose(lm_head_init_rescale, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return

    original_init_weights = model.init_weights

    def patched_init_weights(self, *args, **kwargs):
        generator = original_init_weights(*args, **kwargs)
        with torch.no_grad():
            self.lm_head.w_out.weight.mul_(lm_head_init_rescale)
            if self.lm_head.w_out.bias is not None:
                self.lm_head.w_out.bias.mul_(lm_head_init_rescale)
        return generator

    model.init_weights = MethodType(patched_init_weights, model)


def build_depth_width_model(
    config: base.ExperimentConfig,
    scaling: DepthWidthScalingSpec,
) -> Transformer:
    model = config.model.build(init_device="meta")
    _patch_model_init(model, lm_head_init_rescale=scaling.lm_head_init_rescale)
    _patch_lm_head_forward(model.lm_head, scale=scaling.lm_head_forward_scale)
    return model


def train_with_depth_width_scaling(
    config: base.ExperimentConfig,
    scaling: DepthWidthScalingSpec,
) -> None:
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = build_depth_width_model(config, scaling)
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if get_rank() == 0:
        base.maybe_copy_tokenizer(
            tokenizer_name_or_path=str(config.dataset.tokenizer.identifier),
            save_folder=trainer.save_folder,
        )

    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        base.log.info("Loading checkpoint from %s", config.load_path)
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)

    trainer.fit()


def build_config(
    args: argparse.Namespace,
    overrides: List[str],
) -> Tuple[base.ExperimentConfig, Optional[DepthWidthScalingSpec]]:
    config = base.build_config(args, overrides)
    block_type, alpha = _resolve_parameterization(args.parameterization)
    scaling_axis = _resolve_scaling_axis(args.scaling_axis)

    if scaling_axis == "depth_only":
        return apply_depth_only_scaling(config, args, block_type=block_type, alpha=alpha), None

    return apply_depth_width_scaling(config, args, block_type=block_type, alpha=alpha)


def main() -> None:
    parser = add_preln_mup_args(base.build_parser())
    args, overrides = parser.parse_known_args()
    config, depth_width_scaling = build_config(args, overrides)

    base.prepare_training_environment()
    try:
        if depth_width_scaling is None:
            base.train(config)
        else:
            train_with_depth_width_scaling(config, depth_width_scaling)
    finally:
        base.teardown_training_environment()


if __name__ == "__main__":
    main()
