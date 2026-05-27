"""Load PyTorch chunk files produced by main.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def list_chunks(output_dir: str | Path, split: str) -> list[Path]:
    split_dir = Path(output_dir) / split
    return sorted(split_dir.glob(f"{split}_*.pt"))


def parse_chunk_range(path: Path) -> tuple[int, int] | None:
    """Parse inclusive index range from filename like train_0_999.pt."""
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return int(parts[-2]), int(parts[-1])
    return None


def load_chunk(pt_path: str | Path) -> dict:
    """Load one .pt chunk file."""
    data = torch.load(pt_path, weights_only=False)
    required = {"texts", "labels", "models", "indices"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"{pt_path} missing keys: {sorted(missing)}")
    return data


def chunks_in_range(
    output_dir: str | Path,
    split: str,
    from_idx: int | None = None,
    to_idx: int | None = None,
) -> list[Path]:
    """Return chunk files whose index range overlaps [from_idx, to_idx)."""
    chunks = list_chunks(output_dir, split)
    if from_idx is None and to_idx is None:
        return chunks

    selected = []
    for path in chunks:
        parsed = parse_chunk_range(path)
        if parsed is None:
            selected.append(path)
            continue
        start, end = parsed
        chunk_from = start
        chunk_to = end + 1  # filename end is inclusive
        if from_idx is not None and chunk_to <= from_idx:
            continue
        if to_idx is not None and chunk_from >= to_idx:
            continue
        selected.append(path)
    return selected


class MixedTextChunkDataset(Dataset):
    """PyTorch Dataset for one .pt chunk file."""

    def __init__(self, pt_path: str | Path, from_idx: int | None = None, to_idx: int | None = None):
        self.path = Path(pt_path)
        self.data = load_chunk(self.path)
        self.offsets = list(range(len(self.data["texts"])))

        if from_idx is not None or to_idx is not None:
            filtered = []
            for i in self.offsets:
                global_idx = int(self.data["indices"][i])
                if from_idx is not None and global_idx < from_idx:
                    continue
                if to_idx is not None and global_idx >= to_idx:
                    continue
                filtered.append(i)
            self.offsets = filtered

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        i = self.offsets[index]
        return {
            "text": self.data["texts"][i],
            "labels": self.data["labels"][i],
            "model": self.data["models"][i],
            "index": int(self.data["indices"][i]),
        }


def load_split(
    output_dir: str | Path,
    split: str,
    from_idx: int | None = None,
    to_idx: int | None = None,
) -> ConcatDataset:
    """Load one or more chunks for a split as a single ConcatDataset."""
    paths = chunks_in_range(output_dir, split, from_idx, to_idx)
    if not paths:
        raise FileNotFoundError(
            f"No chunk files for split '{split}' in {output_dir} "
            f"(range [{from_idx}, {to_idx}))"
        )
    return ConcatDataset(
        MixedTextChunkDataset(path, from_idx, to_idx) for path in paths
    )


def collate_mixed_batch(batch: list[dict]) -> dict:
    """Pad variable-length label tensors for DataLoader."""
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-1,
    )
    return {
        "text": [item["text"] for item in batch],
        "labels": labels,
        "model": [item["model"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


def make_dataloader(
    output_dir: str | Path,
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    from_idx: int | None = None,
    to_idx: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = load_split(output_dir, split, from_idx, to_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_mixed_batch,
    )


def words_start_equal(text: str, words: list[str], count: int | None = None) -> bool:
    count = count or min(20, len(words))
    return text.split()[:count] == words[:count]


def find_filtered_index_by_human_start(
    data_dir: str | Path,
    split: str,
    human_start_words: list[str],
    min_words: int = 3,
    hint_index: int | None = None,
    search_radius: int = 2000,
) -> tuple[int, str]:
    """Find the filtered-stream row whose opening words match the mixed human_start."""
    if not human_start_words:
        raise ValueError("human_start_words is empty")

    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    seen = 0
    scan_start = max(0, (hint_index or 0) - search_radius)
    scan_end = (hint_index or 0) + search_radius if hint_index is not None else None

    for file_path in files:
        table = pq.read_table(file_path, columns=["text"])
        for text in table.column("text").to_pylist():
            if not text or not str(text).strip():
                continue
            text = str(text)
            if len(text.split()) < min_words:
                continue
            if seen >= scan_start and words_start_equal(text, human_start_words):
                return seen, text
            seen += 1
            if scan_end is not None and seen > scan_end:
                break
    raise IndexError("No MiniPile text matched the mixed sample human_start prefix.")


def lookup_original_for_sample(
    item: dict,
    data_dir: str | Path,
    split: str,
    chunk_data: dict | None = None,
    default_min_words: int = 3,
    human_start_words: list[str] | None = None,
) -> tuple[str, int, int, str]:
    """
    Resolve the source MiniPile text for a mixed sample.

    Uses chunk metadata when present; falls back to prefix search if the stored
    index no longer maps to the same text (e.g. older chunks or min_words mismatch).
    """
    chunk_data = chunk_data or {}
    stored_index = int(item["index"])
    min_words = int(chunk_data.get("min_words_filter", default_min_words))
    lookup_mode = "index"
    prefix_words = human_start_words or item["text"].split()[:20]

    try:
        original = get_filtered_minipile_text(
            data_dir, split, stored_index, min_words=min_words
        )
    except IndexError:
        original = ""
    else:
        if words_start_equal(original, prefix_words):
            return original, stored_index, min_words, lookup_mode

    human_start = prefix_words
    for candidate_min_words in sorted({min_words, default_min_words, 3, 15, 30}):
        try:
            resolved_index, original = find_filtered_index_by_human_start(
                data_dir,
                split,
                human_start,
                min_words=candidate_min_words,
                hint_index=stored_index,
            )
        except IndexError:
            continue
        lookup_mode = f"prefix_search (stored index {stored_index} pointed elsewhere)"
        return original, resolved_index, candidate_min_words, lookup_mode

    raise IndexError(
        f"Could not find MiniPile source for sample index {stored_index}."
    )


def get_filtered_minipile_text(
    data_dir: str | Path,
    split: str,
    index: int,
    min_words: int = 3,
) -> str:
    """
    Fetch text by filtered-stream index (same rules as utils.load_split_texts).

    Skips empty rows and texts shorter than min_words, so index N matches
    the Nth row that main.py would use, not the Nth raw parquet row.
    """
    if index < 0:
        raise IndexError(f"Index must be >= 0, got {index}")

    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    seen = 0
    for file_path in files:
        table = pq.read_table(file_path, columns=["text"])
        for text in table.column("text").to_pylist():
            if not text or not str(text).strip():
                continue
            if len(str(text).split()) < min_words:
                continue
            if seen == index:
                return str(text)
            seen += 1
    raise IndexError(
        f"Filtered index {index} out of range for split '{split}' "
        f"(only {seen} usable texts found)."
    )


def summarize_chunk(pt_path: str | Path) -> dict:
    data = load_chunk(pt_path)
    label_lengths = [int(t.numel()) for t in data["labels"]]
    text_lengths = [len(text.split()) for text in data["texts"]]
    return {
        "path": str(pt_path),
        "samples": len(data["texts"]),
        "index_range": (
            int(data["indices"][0]),
            int(data["indices"][-1]),
        ),
        "avg_words": sum(text_lengths) / len(text_lengths),
        "avg_labels": sum(label_lengths) / len(label_lengths),
        "models": sorted(set(data["models"])),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Load and inspect .pt chunk files.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a .pt file, or output directory when using --split.",
    )
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--from", dest="from_idx", type=int, default=None)
    parser.add_argument("--to", dest="to_idx", type=int, default=None)
    parser.add_argument("--sample", type=int, default=1, help="Print N sample rows.")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.path and Path(args.path).suffix == ".pt":
        paths = [Path(args.path)]
    elif args.split:
        paths = chunks_in_range(args.output_dir, args.split, args.from_idx, args.to_idx)
    elif args.path:
        paths = list_chunks(args.path, args.split or "train")
    else:
        raise SystemExit("Provide a .pt file path or use --split.")

    if not paths:
        raise SystemExit("No chunk files found.")

    total = 0
    for path in paths:
        info = summarize_chunk(path)
        total += info["samples"]
        print(f"{info['path']}")
        print(f"  samples: {info['samples']}")
        print(f"  indices: {info['index_range'][0]}-{info['index_range'][1]}")
        print(f"  avg words: {info['avg_words']:.1f}")
        print(f"  models: {', '.join(info['models'][:5])}")
        if len(info["models"]) > 5:
            print(f"    ... and {len(info['models']) - 5} more")

        if args.sample > 0:
            ds = MixedTextChunkDataset(path)
            for i in range(min(args.sample, len(ds))):
                item = ds[i]
                preview = item["text"][:120].replace("\n", " ")
                print(f"  [{i}] index={item['index']} model={item['model']}")
                print(f"      text: {preview}...")
                print(f"      labels: {item['labels'].tolist()[:20]}...")

    print(f"\nTotal samples: {total}")


if __name__ == "__main__":
    main()
