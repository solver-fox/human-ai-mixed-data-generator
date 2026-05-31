import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "spm.model",
    "special_tokens_map.json",
    "added_tokens.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate custom token checkpoint to HuggingFace bundle."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("output/dactyl_token_finetuned/best"),
        help="Directory containing token_classifier_state.pt and token_classifier_meta.json",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=Path("output/dactyl_token_finetuned/best_hf_migrated"),
        help="Output directory for HuggingFace-formatted model files.",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=Path("models/DACTYL"),
        help="Fallback base model directory if meta has invalid path.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load the saved HF model after export (uses extra RAM).",
    )
    return parser.parse_args()


def copy_tokenizer_files(src_dir: Path, dst_dir: Path, fallback_dir: Path) -> None:
    for name in TOKENIZER_FILES:
        src = src_dir / name
        if not src.exists():
            src = fallback_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def migrate_low_memory(
    src_weights: Path,
    dst_dir: Path,
    base_model_dir: Path,
    num_labels: int,
) -> None:
    """Write weights/config/tokenizer without instantiating a full model in RAM."""
    cfg = AutoConfig.from_pretrained(base_model_dir)
    cfg.num_labels = num_labels
    cfg.id2label = {0: "human", 1: "ai"}
    cfg.label2id = {"human": 0, "ai": 1}

    dst_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(dst_dir)

    print("Loading checkpoint (mmap)...")
    state = torch.load(
        src_weights,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )

    model_path = dst_dir / "model.safetensors"
    print(f"Writing {model_path} ...")
    save_file(state, model_path)
    del state
    gc.collect()

    copy_tokenizer_files(src_dir=src_weights.parent, dst_dir=dst_dir, fallback_dir=base_model_dir)
    print("Copied tokenizer files.")


def verify_export(dst_dir: Path) -> None:
    from transformers import AutoModelForTokenClassification

    print("Verifying export (loads full model)...")
    model = AutoModelForTokenClassification.from_pretrained(dst_dir, device_map="cpu")
    del model
    gc.collect()


def main():
    args = parse_args()
    project_dir = Path.cwd()

    src_dir = (project_dir / args.src_dir).resolve()
    dst_dir = (project_dir / args.dst_dir).resolve()
    base_fallback = (project_dir / args.base_model_dir).resolve()

    src_weights = src_dir / "token_classifier_state.pt"
    src_meta = src_dir / "token_classifier_meta.json"

    if not src_weights.exists():
        raise FileNotFoundError(f"Missing source weights: {src_weights}")
    if not base_fallback.joinpath("config.json").exists():
        raise FileNotFoundError(f"Invalid fallback base model dir: {base_fallback}")

    if src_meta.exists():
        with open(src_meta, "r") as f:
            meta = json.load(f)
        base_model_dir = Path(meta.get("base_model_dir", str(base_fallback))).resolve()
        num_labels = int(meta.get("num_labels", 2))
    else:
        base_model_dir = base_fallback
        num_labels = 2

    if not base_model_dir.joinpath("config.json").exists():
        print(f"Warning: invalid base model in meta: {base_model_dir}")
        base_model_dir = base_fallback

    print(f"Source checkpoint: {src_weights}")
    print(f"Base model (config only): {base_model_dir}")
    print(f"Destination: {dst_dir}")
    print("Mode: low-memory (direct safetensors write, no duplicate model load)")

    migrate_low_memory(
        src_weights=src_weights,
        dst_dir=dst_dir,
        base_model_dir=base_model_dir,
        num_labels=num_labels,
    )

    print(f"Migrated model saved to: {dst_dir}")
    print("Files:")
    for path in sorted(dst_dir.iterdir()):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {path.name} ({size_mb:.1f} MB)")

    if args.verify:
        verify_export(dst_dir)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
