"""Fast mixed/sandwich window extraction for prepare_data (CPU multiprocessing)."""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm


@dataclass(frozen=True, slots=True)
class ParsedSample:
    words: tuple[str, ...]
    labels: tuple[int, ...]
    boundaries: tuple[int, ...]


def parse_sample(text: str, labels: list[int]) -> ParsedSample | None:
    words = text.split()
    if len(words) != len(labels):
        return None
    n = len(words)
    if n < 2:
        return None

    boundaries = tuple(
        i for i in range(n - 1) if labels[i] != labels[i + 1]
    )
    if not boundaries:
        return None

    return ParsedSample(
        tuple(words),
        tuple(int(x) for x in labels),
        boundaries,
    )


def _word_count_span(word_start: int, word_end: int) -> int:
    return word_end - word_start


def _window_dict(parsed: ParsedSample, start: int, end: int) -> dict:
    words = parsed.words[start:end]
    return {"text": " ".join(words), "label": list(parsed.labels[start:end])}


def _span_covers_boundary(boundary: int, word_start: int, word_end: int) -> bool:
    """True if exclusive span [word_start, word_end) includes words on both sides of boundary."""
    return word_start <= boundary and word_end >= boundary + 2


def _valid_mixed_span(
    parsed: ParsedSample,
    word_start: int,
    word_end: int,
    min_words: int,
    max_words: int,
    boundary: int,
) -> bool:
    if word_end <= word_start or not _span_covers_boundary(boundary, word_start, word_end):
        return False
    n_words = _word_count_span(word_start, word_end)
    if n_words < min_words or n_words > max_words:
        return False
    labels = parsed.labels[word_start:word_end]
    if 0 not in labels or 1 not in labels:
        return False
    return any(labels[i] != labels[i + 1] for i in range(len(labels) - 1))


def sample_window_span(
    parsed: ParsedSample,
    rng: random.Random,
    min_words: int,
    max_words: int,
) -> tuple[int, int] | None:
    """Return (word_start, word_end) spanning a label boundary; end is exclusive."""
    b = parsed.boundaries[rng.randrange(len(parsed.boundaries))]
    n = len(parsed.words)
    if b + 2 > n:
        return None

    word_start = rng.randint(0, b)
    word_end = rng.randint(b + 2, n)
    n_words = _word_count_span(word_start, word_end)

    while n_words > max_words and word_end - word_start > 2:
        can_trim_start = word_start < b
        can_trim_end = word_end > b + 2
        if can_trim_start and (not can_trim_end or rng.random() < 0.5):
            word_start += 1
        elif can_trim_end:
            word_end -= 1
        else:
            break
        n_words = _word_count_span(word_start, word_end)

    while n_words < min_words:
        grew = False
        if word_start > 0:
            word_start -= 1
            grew = True
            n_words = _word_count_span(word_start, word_end)
        if n_words < min_words and word_end < n:
            word_end += 1
            grew = True
            n_words = _word_count_span(word_start, word_end)
        if not grew:
            return None

    if not _valid_mixed_span(parsed, word_start, word_end, min_words, max_words, b):
        return None
    return word_start, word_end


def windows_for_sample(
    text: str,
    labels: list[int],
    rng: random.Random,
    copies: int,
    min_words: int,
    max_words: int,
) -> list[dict]:
    parsed = parse_sample(text, labels)
    if parsed is None:
        return []

    seen: set[tuple[int, int]] = set()
    spans: list[tuple[int, int]] = []
    max_attempts = max(copies * 6, copies)

    for _ in range(max_attempts):
        if len(spans) >= copies:
            break
        span = sample_window_span(parsed, rng, min_words, max_words)
        if span is None or span in seen:
            continue
        s, e = span
        n_words = e - s
        if not (min_words <= n_words <= max_words):
            continue
        seen.add(span)
        spans.append(span)

    return [_window_dict(parsed, s, e) for s, e in spans]


def _augment_one(args: tuple) -> tuple[list[dict], int]:
    sample, seed, copies, min_words, max_words = args
    rng = random.Random(seed)
    windows = windows_for_sample(
        sample["text"],
        sample["label"],
        rng,
        copies,
        min_words,
        max_words,
    )
    return windows, 0 if windows else 1


def augment_mixed_samples(
    samples: list[dict],
    seed: int,
    copies_per_sample: int = 10,
    min_words: int = 35,
    max_words: int = 350,
    desc: str = "augment",
    n_workers: int = 0,
    chunksize: int = 128,
) -> list[dict]:
    """
    Extract random mixed windows in parallel on CPU.

    GPU is not used: work is random indexing on strings, not tensor math.
    """
    n_source = len(samples)
    target_windows = n_source * copies_per_sample
    workers = n_workers if n_workers > 0 else max(1, (os.cpu_count() or 4) - 1)

    work = [
        (samples[i], seed + i, copies_per_sample, min_words, max_words)
        for i in range(n_source)
    ]

    augmented: list[dict] = []
    skipped = 0

    if workers == 1:
        iterator = work
        pbar = tqdm(iterator, desc=desc, unit="source", total=n_source)
        for item in pbar:
            windows, skip = _augment_one(item)
            skipped += skip
            augmented.extend(windows)
            n_gen = len(augmented)
            win_pct = 100.0 * n_gen / target_windows if target_windows else 0.0
            pbar.set_postfix(
                generated=n_gen,
                target=target_windows,
                win_pct=f"{win_pct:.1f}%",
                skipped=skipped,
                workers=1,
                refresh=False,
            )
        pbar.close()
    else:
        print(f"{desc}: using {workers} CPU workers (chunksize={chunksize})")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pbar = tqdm(
                pool.map(_augment_one, work, chunksize=chunksize),
                desc=desc,
                unit="source",
                total=n_source,
            )
            for windows, skip in pbar:
                skipped += skip
                augmented.extend(windows)
                n_gen = len(augmented)
                win_pct = 100.0 * n_gen / target_windows if target_windows else 0.0
                pbar.set_postfix(
                    generated=n_gen,
                    target=target_windows,
                    win_pct=f"{win_pct:.1f}%",
                    skipped=skipped,
                    workers=workers,
                    refresh=False,
                )
            pbar.close()

    print(
        f"{desc}: {n_source} source -> {len(augmented)} windows "
        f"({100.0 * len(augmented) / target_windows:.1f}% of {target_windows} target); "
        f"skipped {skipped} source samples"
    )
    return augmented


def subsample_pure_words_window(
    text: str,
    rng: random.Random,
    label: int,
    min_words: int = 35,
    max_words: int = 350,
) -> dict | None:
    """Random word window for all-human or all-AI text (validator-style one-class subsample)."""
    words = text.split()
    n = len(words)
    if n < min_words:
        return None

    cnt = rng.randint(min_words, min(max_words, n))
    start = rng.randint(0, n - cnt)
    chunk = words[start : start + cnt]
    return {"text": " ".join(chunk), "label": [label] * len(chunk)}


def _pure_words_one(args: tuple) -> dict | None:
    text, seed, label, min_words, max_words = args
    rng = random.Random(seed)
    return subsample_pure_words_window(text, rng, label, min_words, max_words)


def build_pure_word_samples(
    texts: list[str],
    seed: int,
    label: int = 0,
    min_words: int = 35,
    max_words: int = 350,
    desc: str = "pure word window",
    n_workers: int = 0,
    chunksize: int = 256,
    copies_per_doc: int = 1,
    target_count: int | None = None,
) -> list[dict]:
    """Random 35–350 word windows from pure human/AI docs (multiple windows per doc allowed)."""
    copies = max(1, copies_per_doc)
    work = [
        (t, seed + i * 1000 + k, label, min_words, max_words)
        for i, t in enumerate(texts)
        for k in range(copies)
    ]
    workers = n_workers if n_workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    samples: list[dict] = []
    skipped = 0

    if workers == 1:
        pbar = tqdm(work, desc=desc, unit="window", total=len(work))
        for item in pbar:
            row = _pure_words_one(item)
            if row is None:
                skipped += 1
            else:
                samples.append(row)
                if target_count is not None and len(samples) >= target_count:
                    break
            pbar.set_postfix(kept=len(samples), skipped=skipped, refresh=False)
        pbar.close()
    else:
        print(f"{desc}: using {workers} CPU workers (chunksize={chunksize})")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pbar = tqdm(
                pool.map(_pure_words_one, work, chunksize=chunksize),
                desc=desc,
                unit="window",
                total=len(work),
            )
            for row in pbar:
                if row is None:
                    skipped += 1
                else:
                    samples.append(row)
                if target_count is not None and len(samples) >= target_count:
                    pbar.close()
                    break
                pbar.set_postfix(kept=len(samples), skipped=skipped, refresh=False)
            else:
                pbar.close()

    if target_count is not None:
        samples = samples[:target_count]

    print(
        f"{desc}: {len(texts)} docs × {copies} -> {len(samples)} windows "
        f"(target={target_count}); skipped {skipped} attempts"
    )
    return samples


def _process_text_batch(
    texts: list[str],
    seed_base: int,
    label: int,
    min_words: int,
    max_words: int,
    max_copies_per_doc: int,
    target_count: int,
    samples: list[dict],
) -> tuple[int, int]:
    """Append windows from a batch of texts; return (added, skipped_attempts)."""
    added = 0
    skipped = 0
    for i, text in enumerate(texts):
        if len(samples) >= target_count:
            break
        for k in range(max_copies_per_doc):
            if len(samples) >= target_count:
                break
            row = subsample_pure_words_window(
                text,
                random.Random(seed_base + i * 1000 + k),
                label,
                min_words,
                max_words,
            )
            if row is None:
                skipped += 1
            else:
                samples.append(row)
                added += 1
    return added, skipped


def _texts_from_batch(
    batch,
    text_column: str,
    row_label_column: str | None,
    row_label_value: int | None,
) -> list[str]:
    text_idx = batch.schema.get_field_index(text_column)
    text_arr = batch.column(text_idx)
    if row_label_column is None:
        return [t for t in text_arr.to_pylist() if t]

    label_idx = batch.schema.get_field_index(row_label_column)
    label_arr = batch.column(label_idx)
    texts: list[str] = []
    for i in range(batch.num_rows):
        if label_arr[i].as_py() != row_label_value:
            continue
        t = text_arr[i].as_py()
        if t:
            texts.append(t)
    return texts


def stream_pure_word_windows_from_parquet(
    parquet_paths: list[str | Path],
    seed: int,
    target_count: int,
    label: int = 0,
    min_words: int = 35,
    max_words: int = 350,
    max_copies_per_doc: int = 8,
    batch_rows: int = 2048,
    desc: str = "human word window",
    text_column: str = "text",
    row_label_column: str | None = None,
    row_label_value: int | None = None,
) -> list[dict]:
    """
    Stream parquet files → word windows without loading all docs.

    Uses batched parquet reads and single-process processing to avoid RAM/swap spikes
    from giant lists and ProcessPoolExecutor pickling.

    Set row_label_column / row_label_value to keep only matching rows (e.g. label==1).
    """
    import pyarrow.parquet as pq

    paths = [Path(p) for p in parquet_paths]
    if not paths:
        raise FileNotFoundError("No parquet paths provided")

    read_columns = [text_column]
    if row_label_column is not None:
        if row_label_column not in read_columns:
            read_columns.append(row_label_column)
        if row_label_value is None:
            raise ValueError("row_label_value required when row_label_column is set")

    rng = random.Random(seed)
    path_order = paths.copy()
    rng.shuffle(path_order)

    samples: list[dict] = []
    skipped = 0
    docs_seen = 0

    pbar = tqdm(total=target_count, desc=desc, unit="window")
    for path in path_order:
        if len(samples) >= target_count:
            break
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_rows, columns=read_columns
        ):
            if len(samples) >= target_count:
                break
            texts = _texts_from_batch(
                batch, text_column, row_label_column, row_label_value
            )
            docs_seen += len(texts)
            added, batch_skipped = _process_text_batch(
                texts,
                seed + docs_seen,
                label,
                min_words,
                max_words,
                max_copies_per_doc,
                target_count,
                samples,
            )
            skipped += batch_skipped
            pbar.n = len(samples)
            pbar.refresh()
            pbar.set_postfix(docs=docs_seen, skipped=skipped, refresh=False)
    pbar.close()

    samples = samples[:target_count]
    print(
        f"{desc}: {docs_seen} docs scanned -> {len(samples)} windows "
        f"(target={target_count}); skipped {skipped} attempts"
    )
    return samples
