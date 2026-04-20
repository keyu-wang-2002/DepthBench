from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from olmo_core.data.parquet import DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE, iter_tokenized_parquet_batches
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.train.common import MetricMergeStrategy

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ParquetLMEvalCallback(Callback):
    eval_parquet_path: str
    text_field: str
    tokenizer_name_or_path: str
    tokenizer_config: TokenizerConfig
    sequence_length: int
    eval_interval: Optional[int] = 400
    eval_max_batches: int = -1
    shuffle_buffer_size: int = DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE

    def post_step(self):
        if self.step <= 1 or self.eval_interval is None:
            return
        if self.step % self.eval_interval == 0:
            self.perform_eval()

    def post_train(self):
        self.perform_eval(prefix="eval_final")

    def perform_eval(self, prefix: str = "eval"):
        dp_world_size = get_world_size(self.trainer.dp_process_group)
        dp_rank = get_rank(self.trainer.dp_process_group)
        rank_batch_size_tokens = self.trainer.train_module.eval_batch_spec.rank_batch_size
        batch_size_in_sequences = max(1, rank_batch_size_tokens // self.sequence_length)

        total_loss = torch.tensor(0.0, device=self.trainer.device)
        total_tokens = torch.tensor(0.0, device=self.trainer.device)
        num_batches = 0

        log.info("Running parquet LM eval on '%s'...", self.eval_parquet_path)
        for batch in iter_tokenized_parquet_batches(
            parquet_path_or_glob=self.eval_parquet_path,
            text_field=self.text_field,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer_config=self.tokenizer_config,
            sequence_length=self.sequence_length,
            batch_size_in_sequences=batch_size_in_sequences,
            rank=dp_rank,
            world_size=dp_world_size,
            shuffle=False,
            shuffle_seed=0,
            shuffle_buffer_size=self.shuffle_buffer_size,
            include_partial_batch=True,
        ):
            num_batches += 1
            batch = {k: v.to(self.trainer.device) for k, v in batch.items()}
            labels = get_labels(
                batch, label_ignore_index=self.trainer.train_module.label_ignore_index
            )
            valid_mask = labels != self.trainer.train_module.label_ignore_index

            with torch.no_grad():
                output = self.trainer.train_module.eval_batch(batch, labels=labels)

            ce_loss = output.ce_loss
            if ce_loss.shape != labels.shape:
                ce_loss = ce_loss.view_as(labels)

            total_loss += ce_loss.masked_select(valid_mask).sum()
            total_tokens += valid_mask.sum()

            if self.eval_max_batches > 0 and num_batches >= self.eval_max_batches:
                break

        if dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=self.trainer.dp_process_group)
            dist.all_reduce(
                total_tokens, op=dist.ReduceOp.SUM, group=self.trainer.dp_process_group
            )
            num_batches_tensor = torch.tensor(float(num_batches), device=self.trainer.device)
            dist.all_reduce(
                num_batches_tensor,
                op=dist.ReduceOp.SUM,
                group=self.trainer.dp_process_group,
            )
            total_eval_batches = int(num_batches_tensor.item())
        else:
            total_eval_batches = num_batches

        if total_tokens.item() == 0:
            log.warning("LM eval skipped because no valid eval tokens were produced")
            return

        avg_ce_loss = total_loss / total_tokens
        ppl = torch.exp(avg_ce_loss)

        self.trainer.record_metric(
            "lm/CE loss",
            avg_ce_loss,
            reduce_type=None,
            namespace=prefix,
            merge_strategy=MetricMergeStrategy.latest,
        )
        self.trainer.record_metric(
            "lm/PPL",
            ppl,
            reduce_type=None,
            namespace=prefix,
            merge_strategy=MetricMergeStrategy.latest,
        )
        self.trainer.record_metric(
            "lm/batches",
            float(total_eval_batches),
            reduce_type=None,
            namespace=prefix,
            merge_strategy=MetricMergeStrategy.latest,
        )

        ppl_value = ppl.item()
        log.info(
            "Finished parquet LM eval: CE loss=%.4f, PPL=%s, batches=%d",
            avg_ce_loss.item(),
            f"{ppl_value:.4f}" if math.isfinite(ppl_value) else "inf",
            total_eval_batches,
        )
