"""Dataset helpers: batched pretokenization + cached PyTorch Dataset."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def align_labels_for_encoding(word_labels: list[int], word_ids: list[int | None]) -> list[int]:
    aligned: list[int] = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != prev_word_id:
            aligned.append(int(word_labels[word_id]))
        else:
            aligned.append(-100)
        prev_word_id = word_id
    return aligned


def pretokenize_batch(
    word_lists: list[list[str]],
    label_lists: list[list[int]],
    tokenizer,
    max_length: int,
) -> list[dict]:
    if len(word_lists) != len(label_lists):
        raise ValueError("word_lists and label_lists length mismatch")
    for words, labels in zip(word_lists, label_lists):
        if len(words) != len(labels):
            raise ValueError(f"Word/label mismatch: {len(words)} vs {len(labels)}")

    enc = tokenizer(
        word_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    rows = []
    for j in range(len(word_lists)):
        rows.append(
            {
                "input_ids": enc["input_ids"][j],
                "attention_mask": enc["attention_mask"][j],
                "labels": align_labels_for_encoding(label_lists[j], enc.word_ids(batch_index=j)),
            }
        )
    return rows


def pretokenize_split(
    name: str,
    samples: list[dict],
    tokenizer,
    max_length: int,
    cache_path: Path,
    *,
    batch_size: int = 512,
    rebuild_cache: bool = False,
) -> list[dict]:
    """Tokenize once and cache to disk. Batched for speed; tqdm writes to stdout."""
    if cache_path.exists() and not rebuild_cache:
        print(f"[{name}] Loading cache ({cache_path.name})", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    n = len(samples)
    print(
        f"[{name}] Building cache: {cache_path.name} | {n:,} samples | "
        f"batch_size={batch_size} | max_length={max_length}",
        flush=True,
    )
    t0 = time.perf_counter()
    rows: list[dict] = []

    # Text tqdm bar (reliable in VS Code / Jupyter; not ipywidgets)
    pbar = tqdm(
        total=n,
        desc=f"pretokenize {name}",
        unit="sample",
        file=sys.stdout,
        mininterval=0.3,
        dynamic_ncols=True,
    )

    for start in range(0, n, batch_size):
        chunk = samples[start : start + batch_size]
        word_lists = [s["text"].split() for s in chunk]
        label_lists = [s["label"] for s in chunk]
        rows.extend(pretokenize_batch(word_lists, label_lists, tokenizer, max_length))
        done = min(start + batch_size, n)
        pbar.update(done - pbar.n)
        if done % (batch_size * 20) == 0 or done == n:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            tqdm.write(
                f"  [{name}] {done:,}/{n:,} ({100 * done / n:.1f}%) "
                f"| {rate:.0f} samples/s | elapsed {elapsed / 60:.1f} min"
            )

    pbar.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(rows, f)

    elapsed = time.perf_counter() - t0
    print(
        f"[{name}] Cached {len(rows):,} rows in {elapsed / 60:.1f} min "
        f"({len(rows) / elapsed:.0f} samples/s) -> {cache_path}",
        flush=True,
    )
    return rows


class CachedTokenDataset(Dataset):
    """Pre-tokenized rows; no tokenizer in __getitem__."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


def make_token_collator(tokenizer):
    pad_id = tokenizer.pad_token_id or 0

    def collator(batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collator


def training_limits(fast_train: bool) -> dict:
    if fast_train:
        return {
            "MAX_LENGTH": 128,
            "TRAIN_LIMIT": 100_000,
            "VAL_LIMIT": 5_000,
            "TEST_LIMIT": 10_000,
            "TRAIN_BATCH_SIZE": 16,
            "EVAL_BATCH_SIZE": 8,
            "GRAD_ACCUM_STEPS": 4,
            "NUM_EPOCHS": 1,
            "EVAL_STRATEGY": "steps",
            "EVAL_STEPS": 200,
            "SAVE_STRATEGY": "steps",
            "SAVE_STEPS": 200,
            "PRETOKENIZE_BATCH_SIZE": 1024,
        }
    return {
        "MAX_LENGTH": 256,
        "TRAIN_LIMIT": 500_000,
        "VAL_LIMIT": 10_000,
        "TEST_LIMIT": 50_000,
        "TRAIN_BATCH_SIZE": 16,
        "EVAL_BATCH_SIZE": 8,
        "GRAD_ACCUM_STEPS": 8,
        "NUM_EPOCHS": 3,
        "EVAL_STRATEGY": "epoch",
        "EVAL_STEPS": None,
        "SAVE_STRATEGY": "epoch",
        "SAVE_STEPS": None,
        "PRETOKENIZE_BATCH_SIZE": 512,
    }
