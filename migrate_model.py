import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
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
    return parser.parse_args()


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

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Source checkpoint: {src_weights}")
    print(f"Base model: {base_model_dir}")
    print(f"Destination: {dst_dir}")

    cfg = AutoConfig.from_pretrained(base_model_dir)
    cfg.num_labels = num_labels
    cfg.id2label = {0: "human", 1: "ai"}
    cfg.label2id = {"human": 0, "ai": 1}

    hf_model = AutoModelForTokenClassification.from_config(cfg).cpu()
    base_seq = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir, device_map="cpu"
    )
    hf_model.deberta.load_state_dict(base_seq.deberta.state_dict(), strict=True)

    state = torch.load(src_weights, map_location="cpu")
    load_info = hf_model.load_state_dict(state, strict=False)
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)

    if any(key.startswith("classifier") for key in load_info.missing_keys):
        raise RuntimeError(
            f"Classifier weights missing after load: {load_info.missing_keys}"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(dst_dir)

    tokenizer_src = src_dir if src_dir.joinpath("tokenizer_config.json").exists() else base_model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    tokenizer.save_pretrained(dst_dir)

    print(f"Migrated model saved to: {dst_dir}")
    print("Files:")
    for path in sorted(dst_dir.iterdir()):
        print(f"  - {path.name}")

    del base_seq, hf_model, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
