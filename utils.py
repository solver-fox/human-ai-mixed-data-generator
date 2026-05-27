import argparse
import asyncio
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import torch
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, InternalServerError, NotFoundError, RateLimitError

DEFAULT_CONCURRENCY = 32
DEFAULT_DATA_DIR = "dataset/minipile/data"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_CHUNK_SIZE = 1000
MIN_TEXT_WORDS = 3
MAX_MODEL_RETRIES = int(os.environ.get("OPENROUTER_MAX_RETRIES", "10"))
MAX_RATE_LIMIT_WAITS = int(os.environ.get("OPENROUTER_MAX_RATE_WAITS", "20"))
MAX_ENTRY_RETRIES = int(os.environ.get("OPENROUTER_ENTRY_RETRIES", "8"))
MODELS_API_URL = "https://openrouter.ai/api/v1/models?output_modalities=text"


def get_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY in your environment or .env file."
        )
    return api_key


def get_concurrency(total_entries):
    concurrency = int(os.environ.get("OPENROUTER_CONCURRENCY", DEFAULT_CONCURRENCY))
    if concurrency < 1:
        raise ValueError("OPENROUTER_CONCURRENCY must be at least 1.")
    return min(concurrency, total_entries)


def get_client(api_key, concurrency):
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


def parse_price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def model_passes_cost_filter(model, free_only, max_prompt_price, max_completion_price):
    pricing = model.get("pricing", {})
    prompt_price = parse_price(pricing.get("prompt"))
    completion_price = parse_price(pricing.get("completion"))

    if free_only:
        return ":free" in model["id"] or (prompt_price == 0 and completion_price == 0)

    if max_prompt_price is not None and prompt_price > max_prompt_price:
        return False
    if max_completion_price is not None and completion_price > max_completion_price:
        return False
    return True


def fetch_chat_models(api_key, free_only=False, max_prompt_price=None, max_completion_price=None):
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
        if not model_passes_cost_filter(
            model, free_only, max_prompt_price, max_completion_price
        ):
            continue
        models.append(model_id)

    if not models:
        raise RuntimeError("No chat-capable text models matched your cost filters.")

    return models


def get_models(api_key):
    env_models = os.environ.get("OPENROUTER_MODELS")

    if env_models:
        return [model.strip() for model in env_models.split(",") if model.strip()]

    free_only = os.environ.get("OPENROUTER_FREE_ONLY", "").lower() in ("1", "true", "yes")

    max_prompt_price = os.environ.get("OPENROUTER_MAX_PROMPT_PRICE")
    max_completion_price = os.environ.get("OPENROUTER_MAX_COMPLETION_PRICE")
    return fetch_chat_models(
        api_key,
        free_only=free_only,
        max_prompt_price=float(max_prompt_price) if max_prompt_price else None,
        max_completion_price=float(max_completion_price) if max_completion_price else None,
    )


def load_split_texts(data_dir, split, start=0, end=None, min_words=MIN_TEXT_WORDS):
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
            if len(str(text).split()) < min_words:
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


def to_pytorch_chunk(entries, start_index):
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


def save_pytorch_chunk(split, entries, output_dir, global_start):
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    global_end = global_start + len(entries) - 1
    output_path = split_dir / f"{split}_{global_start}_{global_end}.pt"
    torch.save(to_pytorch_chunk(entries, global_start), output_path)
    return output_path


def save_pytorch_chunks(split, entries, output_dir, from_idx, chunk_size):
    written = []
    for offset in range(0, len(entries), chunk_size):
        chunk = entries[offset : offset + chunk_size]
        written.append(
            save_pytorch_chunk(split, chunk, output_dir, from_idx + offset)
        )
    return written


def parse_retry_after_seconds(exc, default=10.0):
    try:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            metadata = body.get("error", {}).get("metadata", {})
            for key in ("retry_after_seconds", "retry_after_seconds_raw"):
                value = metadata.get(key)
                if value is not None:
                    return min(60.0, float(value) + 1.0)
    except (TypeError, ValueError):
        pass
    return default


def _error_body(exc):
    body = getattr(exc, "body", None)
    return body if isinstance(body, dict) else {}


def _error_message(exc):
    body = _error_body(exc)
    message = body.get("error", {}).get("message", "")
    return message if message else str(exc)


def is_rate_limit_error(exc):
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    return False


def is_daily_quota_error(exc):
    message = _error_message(exc).lower()
    if "daily limit" in message or "limit_rpd" in message:
        return True
    metadata = _error_body(exc).get("error", {}).get("metadata", {})
    headers = metadata.get("headers", {})
    remaining = headers.get("X-RateLimit-Remaining")
    reset = headers.get("X-RateLimit-Reset")
    if remaining is not None and str(remaining) == "0" and reset:
        try:
            reset_seconds = int(reset) / 1000 - time.time()
            if reset_seconds > 3600:
                return True
        except (TypeError, ValueError):
            pass
    return False


def is_temporary_rate_limit(exc):
    if not is_rate_limit_error(exc) or is_daily_quota_error(exc):
        return False
    message = _error_message(exc).lower()
    if "temporarily rate-limited" in message or "retry shortly" in message:
        return True
    metadata = _error_body(exc).get("error", {}).get("metadata", {})
    return any(
        metadata.get(key) is not None
        for key in ("retry_after_seconds", "retry_after_seconds_raw")
    )


def should_retry_with_another_model(exc):
    if isinstance(exc, (ValueError, APIConnectionError, InternalServerError)):
        return True
    if isinstance(exc, (NotFoundError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code in (402, 403, 404, 500, 502, 503):
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


def pick_model(models, failed_models):
    available = [model for model in models if model not in failed_models]
    if not available:
        raise RuntimeError(
            f"All {len(models)} models unavailable (daily limits or repeated errors)."
        )
    return random.choice(available)


def extract_completion_text(response, model):
    if not response.choices:
        raise ValueError(f"Model {model} returned no choices.")
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Model {model} returned empty content.")
    return content.strip()


async def generate_text(client, prompt, words_needed, models, semaphore, failed_models):
    max_tokens = max(64, int(words_needed * 2) + 20)
    last_error = None
    attempts = 0
    rate_limit_waits = 0
    max_attempts = max(MAX_MODEL_RETRIES, len(models))

    while attempts < max_attempts:
        model = pick_model(models, failed_models)
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
            return extract_completion_text(response, model), model
        except Exception as exc:
            if is_temporary_rate_limit(exc):
                if rate_limit_waits >= MAX_RATE_LIMIT_WAITS:
                    raise RuntimeError(
                        f"Rate limit persisted after {MAX_RATE_LIMIT_WAITS} waits."
                    ) from exc
                wait = parse_retry_after_seconds(exc)
                rate_limit_waits += 1
                await asyncio.sleep(wait)
                continue
            if is_rate_limit_error(exc):
                failed_models.add(model)
                attempts += 1
                last_error = exc
                continue
            if not should_retry_with_another_model(exc):
                raise
            failed_models.add(model)
            attempts += 1
            last_error = exc

    raise RuntimeError(
        f"Failed after {max_attempts} model attempts (last model unavailable)."
    ) from last_error


async def run_sequences(
    create_one,
    client,
    entries,
    models,
    concurrency,
    split=None,
    output_dir=None,
    from_idx=0,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    semaphore = asyncio.Semaphore(concurrency)
    failed_models = set()
    total = len(entries)
    generated = 0
    lock = asyncio.Lock()
    results = [None] * total
    saved_chunk_ids = set()

    def is_valid_result(result):
        return (
            isinstance(result, dict)
            and result.get("full_text")
            and result.get("labels")
            and result.get("model")
        )

    def print_progress():
        pct = generated * 100 / total
        print(
            f"\rGenerated: {generated}/{total} ({pct:.1f}%)",
            end="",
            flush=True,
        )

    def save_ready_chunks():
        if split is None or output_dir is None:
            return
        for chunk_id in range((total + chunk_size - 1) // chunk_size):
            if chunk_id in saved_chunk_ids:
                continue
            start = chunk_id * chunk_size
            end = min(start + chunk_size, total)
            chunk = results[start:end]
            if not all(item is not None for item in chunk):
                continue
            path = save_pytorch_chunk(
                split, chunk, output_dir, from_idx + start
            )
            saved_chunk_ids.add(chunk_id)
            for i in range(start, end):
                results[i] = None
            print(f"\nWrote {path}", flush=True)

    async def run_one(index, entry):
        nonlocal generated
        last_error = None
        global_index = from_idx + index

        for entry_attempt in range(MAX_ENTRY_RETRIES):
            try:
                result = await create_one(
                    client, entry, models, semaphore, failed_models
                )
                if not is_valid_result(result):
                    raise ValueError(
                        f"Entry {global_index} did not produce valid generated text."
                    )
                async with lock:
                    results[index] = result
                    generated += 1
                    print_progress()
                    save_ready_chunks()
                return
            except Exception as exc:
                last_error = exc
                if entry_attempt + 1 >= MAX_ENTRY_RETRIES:
                    break
                if is_daily_quota_error(exc) or "all models unavailable" in str(
                    exc
                ).lower():
                    break
                wait = (
                    parse_retry_after_seconds(exc)
                    if is_temporary_rate_limit(exc)
                    else 5 * (entry_attempt + 1)
                )
                print(
                    f"\nEntry {global_index} failed "
                    f"({entry_attempt + 1}/{MAX_ENTRY_RETRIES}), "
                    f"retrying in {wait:.0f}s: {exc.__class__.__name__}",
                    flush=True,
                )
                await asyncio.sleep(wait)

        print(f"\nGiving up on entry {global_index}: {last_error}", flush=True)
        raise last_error

    await asyncio.gather(*(run_one(i, entry) for i, entry in enumerate(entries)))
    save_ready_chunks()
    print()
    return [result for result in results if result is not None]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate human/AI mixed datasets from MiniPile parquet splits."
    )
    parser.add_argument(
        "--mode",
        choices=["append", "sandwitch"],
        default=os.environ.get("GENERATION_MODE", "sandwitch"),
        help="append: human+AI; sandwitch: human+AI+human.",
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
