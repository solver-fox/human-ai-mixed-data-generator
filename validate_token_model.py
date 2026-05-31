#!/usr/bin/env python3
"""Evaluate DACTYL / token-classifier checkpoints with SN32 validator metrics."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

ModelKind = Literal["token", "sequence"]


SN32_FOUNDATION_DIRNAME = "deberta-v3-large-hf-weights"
SN32_WEIGHTS_FILENAME = "deberta-large-ls03-ctx1024.pth"
SN32_DEFAULT_MAX_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DACTYL or token checkpoints with SN32 metrics (fp_score, f1_score, ap_score)."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Model dir: models/DACTYL, llm-detection/models, output/dactyl_token_finetuned/1, etc.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help=(
            "Optional separate classifier weights (.pth). "
            "SN32 miner uses foundation dir + deberta-large-ls03-ctx1024.pth."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("pickles/test_samples.pkl"),
        help="Pickle file with list of {text, label} samples (word-level labels).",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=Path("models/DACTYL"),
        help="Base DACTYL dir for custom .pt checkpoints.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Max tokenizer length (default: read from token_classifier_meta.json or 256).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples (for quick smoke tests).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=("auto", "token", "sequence"),
        help="auto: detect from files; sequence: document-level (models/DACTYL); token: word-aligned head.",
    )
    parser.add_argument(
        "--truncated-fallback",
        type=str,
        default="skip",
        choices=("skip", "human", "ai", "0.5"),
        help="For token models only: how to handle words dropped by truncation.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_model_info(model: nn.Module) -> dict[str, int | str | None]:
    """Summarise HF and custom wrapper models (e.g. DebertaTokenClassifier)."""
    if hasattr(model, "num_labels"):
        num_labels = int(model.num_labels)
    elif hasattr(model, "config"):
        num_labels = int(model.config.num_labels)
    else:
        num_labels = None

    hidden_size = None
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        hidden_size = int(model.config.hidden_size)
    elif hasattr(model, "deberta") and hasattr(model.deberta, "config"):
        hidden_size = int(model.deberta.config.hidden_size)
    elif hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
        hidden_size = int(model.classifier.in_features)

    n_encoder_layers = None
    if hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
        n_encoder_layers = len(model.deberta.encoder.layer)

    return {
        "class_name": model.__class__.__name__,
        "num_labels": num_labels,
        "hidden_size": hidden_size,
        "n_encoder_layers": n_encoder_layers,
    }


class DebertaTokenClassifier(nn.Module):
    """Token-level head on top of DeBERTa encoder (matches 3_training.ipynb)."""

    def __init__(self, backbone_seq_model, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.deberta = backbone_seq_model.deberta
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone_seq_model.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


def sn32_reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict[str, float]]:
    """Same metric logic as llm-detection/detection/validator/reward.py::reward."""
    if len(y_pred) == 0:
        return 0.0, {"fp_score": 0.0, "f1_score": 0.0, "ap_score": 0.0}

    preds = np.round(y_pred)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_true, preds, zero_division=0)
    ap_score = average_precision_score(y_true, y_pred)

    metrics = {
        "fp_score": 1.0 - fp / len(y_pred),
        "f1_score": float(f1),
        "ap_score": float(ap_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_words": int(len(y_pred)),
    }
    reward = sum(metrics[k] for k in ("fp_score", "f1_score", "ap_score")) / 3.0
    metrics["reward"] = float(reward)
    return reward, metrics


def load_samples(path: Path, max_samples: int | None) -> list[dict]:
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def find_sn32_paths(model_dir: Path, weights_path: Path | None) -> tuple[Path, Path] | None:
    """Detect llm-detection SN32 layout: foundation HF dir + sibling .pth weights."""
    if weights_path is not None:
        weights_path = weights_path.resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")
        return model_dir.resolve(), weights_path

    model_dir = model_dir.resolve()
    parent = model_dir.parent

    bundle_foundation = model_dir / SN32_FOUNDATION_DIRNAME
    bundle_weights = model_dir / SN32_WEIGHTS_FILENAME
    if bundle_foundation.joinpath("config.json").exists() and bundle_weights.exists():
        return bundle_foundation.resolve(), bundle_weights.resolve()

    sibling_weights = parent / SN32_WEIGHTS_FILENAME
    if model_dir.joinpath("config.json").exists() and sibling_weights.exists():
        if not (model_dir / "token_classifier_state.pt").exists():
            return model_dir, sibling_weights.resolve()

    if model_dir.name == SN32_FOUNDATION_DIRNAME and model_dir.joinpath("config.json").exists():
        raise FileNotFoundError(
            f"Found SN32 foundation at {model_dir} but missing weights file: {sibling_weights}\n"
            "Download it from llm-detection/docs/mining.md or pass --weights-path explicitly."
        )

    return None


def load_sn32_sequence_model(
    foundation_dir: Path,
    weights_path: Path,
    device: torch.device,
) -> tuple[nn.Module, AutoTokenizer, int, ModelKind]:
    """Load SN32 DebertaClassifier checkpoint (foundation + .pth state_dict)."""
    print(f"SN32 foundation: {foundation_dir}")
    print(f"SN32 weights   : {weights_path}")

    tokenizer = AutoTokenizer.from_pretrained(foundation_dir)
    state = torch.load(weights_path, map_location="cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        foundation_dir,
        state_dict=state,
        attention_probs_dropout_prob=0,
        hidden_dropout_prob=0,
    )
    del state
    model.to(device)
    model.eval()
    return model, tokenizer, SN32_DEFAULT_MAX_LENGTH, "sequence"


def load_model_and_tokenizer(
    model_dir: Path,
    base_model_dir: Path,
    device: torch.device,
    model_type: str = "auto",
    weights_path: Path | None = None,
) -> tuple[nn.Module, AutoTokenizer, int, ModelKind]:
    model_dir = model_dir.resolve()
    base_model_dir = base_model_dir.resolve()

    meta_path = model_dir / "token_classifier_meta.json"
    custom_weights = model_dir / "token_classifier_state.pt"
    hf_weights = model_dir / "model.safetensors"
    hf_bin = model_dir / "pytorch_model.bin"
    config_path = model_dir / "config.json"

    max_length = 256
    num_labels = 2
    kind: ModelKind = "token"

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        max_length = int(meta.get("max_length", max_length))
        num_labels = int(meta.get("num_labels", num_labels))
        meta_base = Path(meta.get("base_model_dir", str(base_model_dir)))
        if meta_base.joinpath("config.json").exists():
            base_model_dir = meta_base.resolve()

    sn32_paths = find_sn32_paths(model_dir, weights_path)
    if sn32_paths is not None and model_type in ("auto", "sequence"):
        foundation_dir, sn32_weights = sn32_paths
        return load_sn32_sequence_model(foundation_dir, sn32_weights, device)

    if model_type == "sequence":
        kind = "sequence"
    elif model_type == "token":
        kind = "token"
    elif custom_weights.exists():
        kind = "token"
    elif config_path.exists():
        cfg = AutoConfig.from_pretrained(model_dir)
        arch = cfg.architectures[0] if cfg.architectures else ""
        if "TokenClassification" in arch:
            kind = "token"
        elif "SequenceClassification" in arch:
            kind = "sequence"
            max_length = int(getattr(cfg, "max_position_embeddings", 512))
        elif hf_weights.exists() or hf_bin.exists():
            kind = "sequence"
            max_length = int(getattr(cfg, "max_position_embeddings", 512))
        else:
            kind = "sequence"
            max_length = int(getattr(cfg, "max_position_embeddings", 512))

    if kind == "sequence":
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json for sequence model: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    elif custom_weights.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        backbone = AutoModelForSequenceClassification.from_pretrained(base_model_dir)
        model = DebertaTokenClassifier(backbone, num_labels=num_labels)
        state = torch.load(custom_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        del backbone, state
    elif hf_weights.exists() or hf_bin.exists() or config_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    else:
        raise FileNotFoundError(
            f"No supported weights in {model_dir}. "
            "Expected token_classifier_state.pt, HF token bundle, or sequence config+weights."
        )

    model.to(device)
    model.eval()
    return model, tokenizer, max_length, kind


class SequenceSampleDataset(Dataset):
    """Document-level models: one score replicated to every word (SN32 miner style)."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = sample["text"]
        words = text.split()
        labels = sample["label"]
        if len(words) != len(labels):
            raise ValueError(f"Word/label mismatch at idx {idx}: {len(words)} vs {len(labels)}")

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "num_words": len(words),
            "labels": np.asarray(labels, dtype=np.int64),
        }


def collate_sequence_batch(batch: list[dict]) -> dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_words": [x["num_words"] for x in batch],
        "labels": [x["labels"] for x in batch],
    }


def sequence_logits_to_doc_prob(logits: torch.Tensor, num_labels: int) -> float:
    """DACTYL uses a single logit (sigmoid); 2-class heads use softmax P(AI)."""
    if num_labels <= 1:
        return float(torch.sigmoid(logits[0]).item())
    probs = torch.softmax(logits, dim=-1)
    return float(probs[1].item())


def predict_sequence_model(
    model: nn.Module,
    samples: list[dict],
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    num_labels = int(model.config.num_labels)
    dataset = SequenceSampleDataset(samples, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequence_batch)

    all_probs: list[float] = []
    all_labels: list[int] = []
    stats = {"samples": len(samples), "skipped_words": 0, "scored_words": 0}

    for batch in tqdm(loader, desc="Predicting (sequence)", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        for i in range(logits.size(0)):
            doc_prob = sequence_logits_to_doc_prob(logits[i], num_labels)
            num_words = batch["num_words"][i]
            labels = batch["labels"][i]

            all_probs.extend([doc_prob] * num_words)
            all_labels.extend(labels.tolist())
            stats["scored_words"] += num_words

    return np.asarray(all_probs, dtype=np.float64), np.asarray(all_labels, dtype=np.int64), stats


def fallback_value(name: str) -> float:
    if name == "human":
        return 0.0
    if name == "ai":
        return 1.0
    return 0.5


class WordSampleDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        words = sample["text"].split()
        labels = sample["label"]
        if len(words) != len(labels):
            raise ValueError(f"Word/label mismatch at idx {idx}: {len(words)} vs {len(labels)}")

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "word_ids": enc.word_ids(),
            "num_words": len(words),
            "labels": np.asarray(labels, dtype=np.int64),
        }


def collate_batch(batch: list[dict]) -> dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_ids": [x["word_ids"] for x in batch],
        "num_words": [x["num_words"] for x in batch],
        "labels": [x["labels"] for x in batch],
    }


def forward_logits(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        if isinstance(model, DebertaTokenClassifier):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs["logits"]
        return model(input_ids=input_ids, attention_mask=attention_mask).logits


def logits_to_word_probs(
    logits: torch.Tensor,
    word_ids: list[int | None],
    num_words: int,
    truncated_fallback: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (word_probs, word_labels_mask, n_skipped) for one sample."""
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    word_probs = np.full(num_words, np.nan, dtype=np.float32)
    seen: set[int] = set()
    for tok_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in seen:
            word_probs[word_id] = probs[tok_idx]
            seen.add(word_id)

    keep_mask = ~np.isnan(word_probs)
    n_skipped = int(num_words - keep_mask.sum())

    if truncated_fallback != "skip" and n_skipped > 0:
        fill = fallback_value(truncated_fallback)
        word_probs[~keep_mask] = fill
        keep_mask = np.ones(num_words, dtype=bool)
        n_skipped = 0

    return word_probs, keep_mask, n_skipped


def predict_all_words(
    model: nn.Module,
    samples: list[dict],
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    truncated_fallback: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dataset = WordSampleDataset(samples, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    all_probs: list[float] = []
    all_labels: list[int] = []
    stats = {"samples": len(samples), "skipped_words": 0, "scored_words": 0}

    for batch in tqdm(loader, desc="Predicting", unit="batch"):
        logits = forward_logits(model, batch, device)

        for i in range(logits.size(0)):
            seq_len = int(batch["attention_mask"][i].sum().item())
            sample_logits = logits[i, :seq_len]
            word_probs, keep_mask, n_skipped = logits_to_word_probs(
                sample_logits,
                batch["word_ids"][i][:seq_len],
                batch["num_words"][i],
                truncated_fallback,
            )
            labels = batch["labels"][i]

            stats["skipped_words"] += n_skipped
            if truncated_fallback == "skip":
                all_probs.extend(word_probs[keep_mask].tolist())
                all_labels.extend(labels[keep_mask].tolist())
                stats["scored_words"] += int(keep_mask.sum())
            else:
                all_probs.extend(word_probs.tolist())
                all_labels.extend(labels.tolist())
                stats["scored_words"] += len(labels)

    return np.asarray(all_probs, dtype=np.float64), np.asarray(all_labels, dtype=np.int64), stats


def predict_text(
    model: nn.Module,
    tokenizer,
    text: str,
    model_kind: ModelKind,
    max_length: int,
    device: torch.device,
    threshold: float = 0.5,
    truncated_fallback: str = "skip",
) -> dict:
    """Run one sentence through the model; return per-word P(AI) and labels."""
    text = text.strip()
    words = text.split()
    if not words:
        raise ValueError("Empty text")

    model.eval()

    if model_kind == "sequence":
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        num_labels = int(model.config.num_labels)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        doc_prob = sequence_logits_to_doc_prob(logits, num_labels)
        probs = np.full(len(words), doc_prob, dtype=np.float64)
        n_skipped = 0
        scored_mask = np.ones(len(words), dtype=bool)
    else:
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)
        logits = forward_logits(
            model,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            device,
        )[0, : int(attention_mask[0].sum().item())]
        probs, scored_mask, n_skipped = logits_to_word_probs(
            logits,
            enc.word_ids()[: logits.size(0)],
            len(words),
            truncated_fallback,
        )
        doc_prob = float(np.nanmean(probs[scored_mask])) if scored_mask.any() else float("nan")

    labels = np.where(probs >= threshold, "ai", "human")
    labels[~scored_mask] = "—"  # truncated / not scored

    return {
        "text": text,
        "words": words,
        "probs": probs,
        "labels": labels.tolist(),
        "scored_mask": scored_mask,
        "doc_prob": doc_prob,
        "model_kind": model_kind,
        "n_skipped": n_skipped,
        "threshold": threshold,
    }


def print_text_predictions(result: dict, max_words: int | None = 80) -> None:
    """Pretty-print per-word predictions from predict_text()."""
    words = result["words"]
    probs = result["probs"]
    labels = result["labels"]
    kind = result["model_kind"]

    print(f"model kind : {kind}")
    if kind == "sequence":
        print(f"document P(AI): {result['doc_prob']:.4f}  (same for every word)")
    if result["n_skipped"]:
        print(f"truncated  : {result['n_skipped']} word(s) not scored (beyond max_length)")

    show = len(words) if max_words is None else min(len(words), max_words)
    print(f"\n{'word':<20} {'P(AI)':>8}  label")
    print("-" * 40)
    for i in range(show):
        p = probs[i]
        p_str = f"{p:.3f}" if not np.isnan(p) else "  n/a"
        print(f"{words[i]:<20} {p_str:>8}  {labels[i]}")

    if show < len(words):
        print(f"... ({len(words) - show} more words)")

    ai_words = sum(1 for l in labels if l == "ai")
    print(f"\nsummary: {ai_words}/{len(words)} words predicted AI (threshold {result['threshold']})")


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  reward   : {metrics['reward']:.4f}")
    print(f"  fp_score : {metrics['fp_score']:.4f}")
    print(f"  f1_score : {metrics['f1_score']:.4f}")
    print(f"  ap_score : {metrics['ap_score']:.4f}")
    print(
        f"  confusion: tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}"
    )
    print(f"  words    : {metrics['n_words']}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    samples = load_samples(args.data.resolve(), args.max_samples)
    model, tokenizer, meta_max_length, model_kind = load_model_and_tokenizer(
        args.model_dir.resolve(),
        args.base_model_dir.resolve(),
        device,
        args.model_type,
        args.weights_path.resolve() if args.weights_path else None,
    )
    max_length = args.max_length if args.max_length is not None else meta_max_length

    print(f"Model dir   : {args.model_dir.resolve()}")
    print(f"Model type  : {model_kind}")
    print(f"Data        : {args.data.resolve()} ({len(samples)} samples)")
    print(f"Device      : {device}")
    print(f"Max length  : {max_length}")
    if model_kind == "token":
        print(f"Truncated   : {args.truncated_fallback}")

    if model_kind == "sequence":
        y_pred, y_true, stats = predict_sequence_model(
            model=model,
            samples=samples,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        y_pred, y_true, stats = predict_all_words(
            model=model,
            samples=samples,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=args.batch_size,
            device=device,
            truncated_fallback=args.truncated_fallback,
        )

    _, metrics = sn32_reward(y_pred, y_true)

    print("\n=== SN32-style metrics (micro, all scored words) ===")
    print_metrics("Overall", metrics)

    if model_kind == "token" and stats["skipped_words"] > 0:
        pct = 100.0 * stats["skipped_words"] / max(stats["scored_words"] + stats["skipped_words"], 1)
        print(
            f"\nTruncation: skipped {stats['skipped_words']} words ({pct:.1f}%) "
            f"outside max_length={max_length}. Use --max-length 512+ or --truncated-fallback 0.5 to include them."
        )

    # Per-class prevalence
    ai_rate = float(y_true.mean()) if len(y_true) else 0.0
    pred_ai_rate = float((y_pred >= 0.5).mean()) if len(y_pred) else 0.0
    print(f"\nLabel AI rate : {ai_rate:.3f}")
    print(f"Pred AI rate  : {pred_ai_rate:.3f} (threshold 0.5)")


if __name__ == "__main__":
    main()
