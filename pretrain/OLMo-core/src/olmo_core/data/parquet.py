from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
import torch.distributed as dist

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_rank, get_world_size

from .collator import DataCollator
from .data_loader import DataLoaderBase, DataLoaderConfig, TextDataLoaderBase
from .tokenizer import TokenizerConfig

import datasets
import datasets.distributed
from transformers import AutoTokenizer

__all__ = [
    "DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE",
    "StreamingParquetDataLoaderConfig",
    "StreamingParquetDataLoader",
    "build_parquet_dataloader",
    "iter_tokenized_parquet_batches",
]

log = logging.getLogger(__name__)


DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE = 10_000



def iter_tokenized_parquet_batches(
    *,
    parquet_path_or_glob: str,
    text_field: str,
    tokenizer_name_or_path: str,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    batch_size_in_sequences: int,
    rank: int,
    world_size: int,
    shuffle: bool,
    shuffle_seed: int,
    shuffle_buffer_size: int = DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE,
    include_partial_batch: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Stream text from parquet, tokenize each row as a document, concatenate the
    resulting token IDs, and yield fixed-length batches.

    This mirrors the main OLMo / OLMo-core pretraining path more closely:
    - each parquet row is treated as one document,
    - each document is tokenized independently and terminated with EOS,
    - token IDs are concatenated into a continuous token stream,
    - the stream is chunked into contiguous blocks of `sequence_length`,
    - leftover tokens that don't fill a full sequence are dropped.

    Setting ``include_partial_batch=True`` allows the final batch to contain fewer
    than ``batch_size_in_sequences`` full sequences, which is useful for eval.
    """

    dataset = datasets.load_dataset(
        "parquet",
        data_files=parquet_path_or_glob,
        split="train",
        streaming=True,
    )
    dataset = datasets.distributed.split_dataset_by_node(
        dataset,
        rank=rank,
        world_size=world_size,
    )
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        model_max_length=sequence_length,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer_config.pad_token_id

    batch_input_ids: List[torch.Tensor] = []
    token_buffer: List[int] = []
    buffer_start = 0

    for example in dataset:
        text = example.get(text_field)
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)

        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if not token_ids:
            token_ids = [tokenizer_config.eos_token_id]
        elif token_ids[-1] != tokenizer_config.eos_token_id:
            token_ids.append(tokenizer_config.eos_token_id)

        token_buffer.extend(token_ids)

        while len(token_buffer) - buffer_start >= sequence_length:
            sequence = torch.tensor(
                token_buffer[buffer_start : buffer_start + sequence_length],
                dtype=torch.long,
            )
            batch_input_ids.append(sequence)
            buffer_start += sequence_length

            if len(batch_input_ids) == batch_size_in_sequences:
                input_ids = torch.stack(batch_input_ids)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
                batch_input_ids = []

            if buffer_start >= sequence_length * 8:
                token_buffer = token_buffer[buffer_start:]
                buffer_start = 0

    if include_partial_batch and batch_input_ids:
        input_ids = torch.stack(batch_input_ids)
        yield {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


@DataLoaderConfig.register("parquet")
@dataclass
class StreamingParquetDataLoaderConfig(DataLoaderConfig["StreamingParquetDataLoader"]):
    """
    Config for a streaming parquet text data loader.
    """

    train_parquet_glob: str
    eval_parquet_path: str
    text_field: str
    tokenizer_name_or_path: str
    tokenizer_config: TokenizerConfig
    sequence_length: int
    global_batch_size: int
    work_dir: str
    seed: int = 34521
    shuffle_buffer_size: int = DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE

    def build(
        self, *, dp_process_group: Optional[dist.ProcessGroup] = None
    ) -> "StreamingParquetDataLoader":
        return StreamingParquetDataLoader(
            train_parquet_glob=self.train_parquet_glob,
            eval_parquet_path=self.eval_parquet_path,
            text_field=self.text_field,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer_config=self.tokenizer_config,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
            work_dir=self.work_dir,
            seed=self.seed,
            shuffle_buffer_size=self.shuffle_buffer_size,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
        )


class StreamingParquetDataLoader(TextDataLoaderBase):
    """
    Stateful OLMo Core data loader backed by a streaming parquet text source.
    """

    def __init__(
        self,
        *,
        train_parquet_glob: str,
        eval_parquet_path: str,
        text_field: str,
        tokenizer_name_or_path: str,
        tokenizer_config: TokenizerConfig,
        sequence_length: int,
        global_batch_size: int,
        work_dir: PathOrStr,
        seed: int = 34521,
        shuffle_buffer_size: int = DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE,
        dp_world_size: int = 1,
        dp_rank: int = 0,
    ):
        super().__init__(
            collator=DataCollator(
                pad_token_id=tokenizer_config.pad_token_id,
                vocab_size=tokenizer_config.padded_vocab_size(),
            ),
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
        if self.rank_batch_size % sequence_length != 0:
            raise ValueError(
                "Per-rank token batch size must be divisible by sequence length, got "
                f"{self.rank_batch_size} and {sequence_length}"
            )

        self.train_parquet_glob = train_parquet_glob
        self.eval_parquet_path = eval_parquet_path
        self.text_field = text_field
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_config = tokenizer_config
        self.sequence_length = sequence_length
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rank_batch_size_in_sequences = self.rank_batch_size // self.sequence_length

    @property
    def total_batches(self) -> Optional[int]:
        return None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in DataLoaderBase.__iter__(self):
            self.tokens_processed += self.global_num_tokens_in_batch(batch) or 0
            yield batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "tokens_processed": self.tokens_processed,
            "seed": self.seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.batches_processed = state_dict.get("batches_processed", 0)
        self.tokens_processed = state_dict.get("tokens_processed", 0)
        self.seed = state_dict.get("seed", self.seed)
        self._epoch = state_dict.get("epoch", self._epoch)

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch

    def get_mock_batch(self) -> Dict[str, Any]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.dp_rank)
        return {
            "input_ids": torch.randint(
                0,
                self.tokenizer_config.vocab_size,
                (self.rank_batch_size_in_sequences, self.sequence_length),
                generator=generator,
            ),
            "attention_mask": torch.ones(
                (self.rank_batch_size_in_sequences, self.sequence_length),
                dtype=torch.long,
            ),
        }

    def global_num_tokens_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        return batch["input_ids"].numel() * self.dp_world_size

    def _iter_rank_batches(self) -> Iterator[Dict[str, Any]]:
        yield from iter_tokenized_parquet_batches(
            parquet_path_or_glob=self.train_parquet_glob,
            text_field=self.text_field,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer_config=self.tokenizer_config,
            sequence_length=self.sequence_length,
            batch_size_in_sequences=self.rank_batch_size_in_sequences,
            rank=self.dp_rank,
            world_size=self.dp_world_size,
            shuffle=True,
            shuffle_seed=self.seed + self.epoch,
            shuffle_buffer_size=self.shuffle_buffer_size,
            include_partial_batch=False,
        )

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        batch_iter = self._iter_rank_batches()

        # If we resume from a checkpoint, skip batches that were already consumed.
        for _ in range(self.batches_processed):
            if next(batch_iter, None) is None:
                return

        yield from batch_iter


def build_parquet_dataloader(
    *,
    train_parquet_glob: str,
    eval_parquet_path: str,
    text_field: str,
    tokenizer_name_or_path: str,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    global_batch_size: int,
    work_dir: str,
    seed: int = 34521,
    shuffle_buffer_size: int = DEFAULT_PARQUET_SHUFFLE_BUFFER_SIZE,
    dp_process_group: Optional[dist.ProcessGroup] = None,
) -> StreamingParquetDataLoader:
    """
    Convenience wrapper for callers that want a parquet data loader directly.
    """
    return StreamingParquetDataLoaderConfig(
        train_parquet_glob=train_parquet_glob,
        eval_parquet_path=eval_parquet_path,
        text_field=text_field,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_config=tokenizer_config,
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        work_dir=work_dir,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    ).build(dp_process_group=dp_process_group)
