import argparse
import asyncio
import json
import os
import random
import urllib.request
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, NotFoundError, RateLimitError

load_dotenv()

DEFAULT_CONCURRENCY = 32
DEFAULT_DATA_DIR = "dataset/minipile/data"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_CHUNK_SIZE = 1000
MIN_TEXT_WORDS = 3
MAX_MODEL_RETRIES = 10
MODELS_API_URL = "https://openrouter.ai/api/v1/models?output_modalities=text"


def _get_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY in your environment or .env file."
        )
    return api_key


def _get_concurrency(total_entries):
    concurrency = int(os.environ.get("OPENROUTER_CONCURRENCY", DEFAULT_CONCURRENCY))
    if concurrency < 1:
        raise ValueError("OPENROUTER_CONCURRENCY must be at least 1.")
    return min(concurrency, total_entries)


def _get_client(api_key, concurrency):
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=concurrency + 10,
            max_keepalive_connections=concurrency,
        ),
        timeout=httpx.Timeout(120.0),
    )
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        http_client=http_client,
    )


def _parse_price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _model_passes_cost_filter(model, free_only, max_prompt_price, max_completion_price):
    pricing = model.get("pricing", {})
    prompt_price = _parse_price(pricing.get("prompt"))
    completion_price = _parse_price(pricing.get("completion"))

    if free_only:
        return ":free" in model["id"] or (prompt_price == 0 and completion_price == 0)

    if max_prompt_price is not None and prompt_price > max_prompt_price:
        return False
    if max_completion_price is not None and completion_price > max_completion_price:
        return False
    return True


def _fetch_chat_models(api_key, free_only=False, max_prompt_price=None, max_completion_price=None):
    request = urllib.request.Request(
        MODELS_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)

    models = []
    for model in payload["data"]:
        model_id = model["id"]
        if model_id.startswith("~"):
            continue
        if "embed" in model_id.lower():
            continue
        if "max_tokens" not in model.get("supported_parameters", []):
            continue
        outputs = model.get("architecture", {}).get("output_modalities", [])
        if "text" not in outputs:
            continue
        if "audio" in outputs:
            continue
        inputs = model.get("architecture", {}).get("input_modalities", [])
        if "text" not in inputs:
            continue
        if model.get("expiration_date"):
            continue
        if not _model_passes_cost_filter(
            model, free_only, max_prompt_price, max_completion_price
        ):
            continue
        models.append(model_id)

    if not models:
        raise RuntimeError("No chat-capable text models matched your cost filters.")

    return models


def _get_models(api_key):
    env_models = os.environ.get("OPENROUTER_MODELS")
    if env_models:
        return [model.strip() for model in env_models.split(",") if model.strip()]

    free_only = os.environ.get("OPENROUTER_FREE_ONLY", "").lower() in ("1", "true", "yes")
    max_prompt_price = os.environ.get("OPENROUTER_MAX_PROMPT_PRICE")
    max_completion_price = os.environ.get("OPENROUTER_MAX_COMPLETION_PRICE")
    return _fetch_chat_models(
        api_key,
        free_only=free_only,
        max_prompt_price=float(max_prompt_price) if max_prompt_price else None,
        max_completion_price=float(max_completion_price) if max_completion_price else None,
    )


def _load_split_texts(data_dir, split, start=0, end=None):
    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' in {data_dir}"
        )
    if start < 0:
        raise ValueError("--from must be >= 0.")
    if end is not None and end <= start:
        raise ValueError("--to must be greater than --from.")

    texts = []
    index = 0
    for file_path in files:
        table = pq.read_table(file_path, columns=["text"])
        for text in table.column("text").to_pylist():
            if not text or not str(text).strip():
                continue
            if len(str(text).split()) < MIN_TEXT_WORDS:
                continue

            if index < start:
                index += 1
                continue
            if end is not None and index >= end:
                return texts

            texts.append(str(text))
            index += 1

    if not texts:
        raise ValueError(
            f"No usable texts found for split '{split}' in range [{start}, {end})."
        )

    return texts


def _to_pytorch_chunk(entries, start_index):
    return {
        "texts": [entry["full_text"] for entry in entries],
        "labels": [
            torch.tensor(entry["labels"], dtype=torch.long) for entry in entries
        ],
        "models": [entry["model"] for entry in entries],
        "indices": torch.arange(
            start_index, start_index + len(entries), dtype=torch.long
        ),
    }


def _save_pytorch_chunks(split, mixed_entries, output_dir, from_idx, chunk_size):
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for offset in range(0, len(mixed_entries), chunk_size):
        chunk = mixed_entries[offset : offset + chunk_size]
        global_start = from_idx + offset
        global_end = global_start + len(chunk) - 1
        filename = f"{split}_{global_start}_{global_end}.pt"
        output_path = split_dir / filename
        torch.save(_to_pytorch_chunk(chunk, global_start), output_path)
        written.append(output_path)

    return written


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate human/AI mixed datasets from MiniPile parquet splits."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MINIPILE_DATA_DIR", DEFAULT_DATA_DIR),
        help="Directory containing MiniPile parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MINIPILE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Directory for mixed train/validation/test PyTorch chunk files.",
    )
    parser.add_argument(
        "--splits",
        default=os.environ.get("MINIPILE_SPLITS", ",".join(DEFAULT_SPLITS)),
        help="Comma-separated splits to process (train,validation,test).",
    )
    parser.add_argument(
        "--from",
        dest="from_idx",
        type=int,
        default=int(os.environ.get("MINIPILE_FROM", "0")),
        help="Start index in each split (inclusive).",
    )
    parser.add_argument(
        "--to",
        dest="to_idx",
        type=int,
        default=int(os.environ["MINIPILE_TO"]) if os.environ.get("MINIPILE_TO") else None,
        help="End index in each split (exclusive).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("MINIPILE_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
        help="Number of texts per output .pt file.",
    )
    return parser.parse_args()


def _prepare_sequence(human_text):
    ai_words_min = int(os.environ.get("OPENROUTER_AI_WORDS_MIN", "10"))
    ai_words_max = int(os.environ.get("OPENROUTER_AI_WORDS_MAX", "100"))
    human_words = human_text.split()
    word_cutoff = random.randint(3, min(100, len(human_words)))
    human_subtext = " ".join(human_words[:word_cutoff])
    words_needed = random.randint(ai_words_min, ai_words_max)
    prompt = (
        f"Continue writing the following text naturally. Generate exactly {words_needed} words, "
        f"so that the final combined text reaches a grand total of {words_needed + word_cutoff} words. "
        f"Do not repeat the prompt or the input text:\n\n{human_subtext}"
    )
    return human_words, word_cutoff, prompt, words_needed


def _should_retry_with_another_model(exc):
    if isinstance(exc, ValueError):
        return True
    if isinstance(exc, (NotFoundError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code in (402, 403, 404, 502, 503):
            return True
        if exc.status_code == 400:
            message = str(exc).lower()
            if "provider returned error" in message:
                return True
            return any(
                marker in message
                for marker in (
                    "model_not_available",
                    "model_not_found",
                    "non-serverless",
                    "not found",
                    "does not exist",
                    "do not have access",
                    "no longer available",
                    "invalid model",
                    "unsupported model",
                    "invalid argument",
                    "invalid_argument",
                )
            )
    return False


def _pick_model(models, failed_models):
    available = [model for model in models if model not in failed_models]
    if not available:
        available = models
    return random.choice(available)


def _build_result(human_words, word_cutoff, ai_append_text, model):
    ai_words = ai_append_text.split()
    all_words = human_words[:word_cutoff] + ai_words
    labels = [0] * word_cutoff + [1] * len(ai_words)

    return {
        "full_text": " ".join(all_words),
        "labels": labels,  # 0 for human, 1 for AI
        "model": model,
    }


def _extract_completion_text(response, model):
    if not response.choices:
        raise ValueError(f"Model {model} returned no choices.")
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Model {model} returned empty content.")
    return content.strip()


async def create_mixed_sequence(client, human_text, models, semaphore, failed_models):
    human_words, word_cutoff, prompt, words_needed = _prepare_sequence(human_text)
    max_tokens = max(64, int(words_needed * 2) + 20)
    last_error = None

    for _ in range(MAX_MODEL_RETRIES):
        model = _pick_model(models, failed_models)
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
            ai_append_text = _extract_completion_text(response, model)
            return _build_result(human_words, word_cutoff, ai_append_text, model)
        except Exception as exc:
            if not _should_retry_with_another_model(exc):
                raise
            failed_models.add(model)
            last_error = exc

    raise RuntimeError(
        f"Failed after {MAX_MODEL_RETRIES} model attempts (last model unavailable)."
    ) from last_error


async def create_mixed_sequences(client, human_entries, models, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    failed_models = set()
    total = len(human_entries)
    completed = 0
    lock = asyncio.Lock()
    results = [None] * total

    async def run_one(index, entry):
        nonlocal completed
        result = await create_mixed_sequence(
            client, entry, models, semaphore, failed_models
        )
        async with lock:
            completed += 1
            results[index] = result
            pct = completed * 100 / total
            print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end="", flush=True)

    await asyncio.gather(*(run_one(i, entry) for i, entry in enumerate(human_entries)))
    print()
    return results


async def main():
    args = _parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("At least one split must be specified.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = _get_api_key()
    models = _get_models(api_key)
    print(f"Using {len(models)} OpenRouter chat models (random per entry).")
    range_label = f"[{args.from_idx}, {args.to_idx})" if args.to_idx is not None else f"[{args.from_idx}, end)"
    print(f"Index range per split: {range_label}, chunk size: {args.chunk_size}")

    for split in splits:
        human_entries = _load_split_texts(
            args.data_dir, split, start=args.from_idx, end=args.to_idx
        )
        concurrency = _get_concurrency(len(human_entries))
        client = _get_client(api_key, concurrency)

        print(f"\n{split}: {len(human_entries)} entries, {concurrency} parallel requests")
        try:
            mixed_entries = await create_mixed_sequences(
                client, human_entries, models, concurrency
            )
        finally:
            await client.close()

        written = _save_pytorch_chunks(
            split,
            mixed_entries,
            output_dir,
            args.from_idx,
            args.chunk_size,
        )
        for path in written:
            print(f"Wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
