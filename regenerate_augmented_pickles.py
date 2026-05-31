#!/usr/bin/env python3
"""Regenerate mixed/sandwich augmented pickles from .pt chunks (fixed subsampler)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from generator.load_pt import list_chunks, load_chunk
from generator.prepare_subsample import augment_mixed_samples

MIN_WINDOW_WORDS = 35
MAX_WINDOW_WORDS = 350
AUGMENT_COPIES = 10
RANDOM_SEED = 42
AUGMENT_WORKERS = 0
AUGMENT_CHUNKSIZE = 128

MIXED_V2_DIR = PROJECT_DIR / "output" / "mixed_v2"
SANDWITCH_V2_DIR = PROJECT_DIR / "output" / "sandwitch_v2"
PICKLE_DIR = PROJECT_DIR / "pickles"


def labels_to_list(labels) -> list[int]:
    if hasattr(labels, "tolist"):
        return [int(x) for x in labels.tolist()]
    return [int(x) for x in labels]


def load_raw_from_dir(output_dir: Path, source_mode: str) -> list[dict]:
    splits = sorted(
        d.name for d in output_dir.iterdir() if d.is_dir() and any(d.glob("*.pt"))
    )
    samples = []
    for source_split in splits:
        for path in list_chunks(output_dir, source_split):
            data = load_chunk(path)
            for i in range(len(data["texts"])):
                samples.append(
                    {
                        "text": data["texts"][i],
                        "label": labels_to_list(data["labels"][i]),
                    }
                )
    print(f"{source_mode}: {len(samples)} raw from {output_dir} ({', '.join(splits)})")
    return samples


def validate_pickles(samples: list[dict], name: str) -> None:
    bad_len = bad_pure = bad_bound = 0
    for s in samples:
        text, labels = s["text"], s["label"]
        n_words = len(text.split())
        if not (MIN_WINDOW_WORDS <= n_words <= MAX_WINDOW_WORDS):
            bad_len += 1
        if not (0 in labels and 1 in labels):
            bad_pure += 1
        if not any(labels[i] != labels[i + 1] for i in range(len(labels) - 1)):
            bad_bound += 1
    print(
        f"  {name} validate: n={len(samples)} bad_len={bad_len} "
        f"bad_pure={bad_pure} bad_bound={bad_bound}"
    )
    if bad_len or bad_pure or bad_bound:
        raise SystemExit(f"{name} validation failed")


def main() -> None:
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)

    mixed_raw = load_raw_from_dir(MIXED_V2_DIR, "mixed_v2")
    sandwich_raw = load_raw_from_dir(SANDWITCH_V2_DIR, "sandwitch_v2")

    with open(PICKLE_DIR / "mixed_samples_raw.pkl", "wb") as f:
        pickle.dump(mixed_raw, f)
    with open(PICKLE_DIR / "sandwich_samples_raw.pkl", "wb") as f:
        pickle.dump(sandwich_raw, f)

    mixed_aug = augment_mixed_samples(
        mixed_raw,
        seed=RANDOM_SEED,
        copies_per_sample=AUGMENT_COPIES,
        min_words=MIN_WINDOW_WORDS,
        max_words=MAX_WINDOW_WORDS,
        desc="mixed augment",
        n_workers=AUGMENT_WORKERS,
        chunksize=AUGMENT_CHUNKSIZE,
    )
    sandwich_aug = augment_mixed_samples(
        sandwich_raw,
        seed=RANDOM_SEED + 1,
        copies_per_sample=AUGMENT_COPIES,
        min_words=MIN_WINDOW_WORDS,
        max_words=MAX_WINDOW_WORDS,
        desc="sandwich augment",
        n_workers=AUGMENT_WORKERS,
        chunksize=AUGMENT_CHUNKSIZE,
    )

    with open(PICKLE_DIR / "mixed_samples.pkl", "wb") as f:
        pickle.dump(mixed_aug, f)
    with open(PICKLE_DIR / "sandwich_samples.pkl", "wb") as f:
        pickle.dump(sandwich_aug, f)

    validate_pickles(mixed_aug, "mixed")
    validate_pickles(sandwich_aug, "sandwich")
    print("Done. pickles/mixed_samples.pkl and pickles/sandwich_samples.pkl updated.")


if __name__ == "__main__":
    main()
