"""Analyze word counts per text, grouped by validator."""

import json
import statistics
from collections import defaultdict
from pathlib import Path


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    req_dir = Path(__file__).parent
    files = sorted(
        p for p in req_dir.glob("*.json")
        if not p.name.startswith(".")
    )

    # validator_uid -> list of (filename, text_index, word_count)
    per_validator: dict[int, list[tuple[str, int, int]]] = defaultdict(list)

    for path in files:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        uid = data["validator_uid"]
        for i, text in enumerate(data.get("texts", [])):
            wc = word_count(text)
            per_validator[uid].append((path.name, i, wc))

    print(f"Analyzed {len(files)} JSON files\n")
    print("=" * 80)

    for uid in sorted(per_validator):
        entries = per_validator[uid]
        counts = [wc for _, _, wc in entries]

        print(f"\nValidator UID: {uid}")
        print(f"  Texts: {len(counts)}  |  "
              f"min: {min(counts)}  max: {max(counts)}  |  "
              f"mean: {statistics.mean(counts):.1f}  median: {statistics.median(counts):.1f}")

        min_entry = min(entries, key=lambda e: e[2])
        max_entry = max(entries, key=lambda e: e[2])
        print(f"  Shortest: {min_entry[2]} words  ({min_entry[0]} idx {min_entry[1]})")
        print(f"  Longest:  {max_entry[2]} words  ({max_entry[0]} idx {max_entry[1]})")
        print()
        print(f"  {'File':<42} {'Texts':>5} {'Min':>5} {'Max':>5}")
        print(f"  {'-'*42} {'-'*5} {'-'*5} {'-'*5}")

        by_file: dict[str, list[int]] = defaultdict(list)
        for name, _, wc in entries:
            by_file[name].append(wc)

        for name in sorted(by_file):
            wcs = by_file[name]
            print(f"  {name:<42} {len(wcs):>5} {min(wcs):>5} {max(wcs):>5}")

    print("\n" + "=" * 80)
    print("\nSummary:\n")
    print(f"  {'Validator':>10} {'Files':>6} {'Texts':>6} {'Min':>5} {'Max':>5} "
          f"{'Mean':>8} {'Median':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*8} {'-'*8}")
    for uid in sorted(per_validator):
        entries = per_validator[uid]
        counts = [wc for _, _, wc in entries]
        file_count = len({name for name, _, _ in entries})
        print(
            f"  {uid:>10} {file_count:>6} {len(counts):>6} "
            f"{min(counts):>5} {max(counts):>5} "
            f"{statistics.mean(counts):>8.1f} {statistics.median(counts):>8.1f}"
        )


if __name__ == "__main__":
    main()
