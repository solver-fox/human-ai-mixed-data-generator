import argparse
import asyncio
import json
import os
import random
import time
import urllib.request
from contextlib import contextmanager
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
MAX_ENTRY_RETRIES = int(os.environ.get("OPENROUTER_ENTRY_RETRIES", "12"))
MAX_RATE_LIMIT_STRIKES = int(os.environ.get("OPENROUTER_RATE_LIMIT_STRIKES", "3"))
MAX_TRANSIENT_RETRIES = int(os.environ.get("OPENROUTER_TRANSIENT_RETRIES", "20"))
TRANSIENT_RETRY_BASE_SECONDS = float(
    os.environ.get("OPENROUTER_TRANSIENT_RETRY_SECONDS", "5")
)
MODELS_API_URL = "https://openrouter.ai/api/v1/models?output_modalities=text"


def timing_enabled():
    return os.environ.get("OPENROUTER_TIMING", "1").lower() in ("1", "true", "yes")


def log_timing_msg(label, seconds):
    if timing_enabled():
        print(f"[timing] {label}: {seconds:.2f}s", flush=True)


@contextmanager
def log_timing(label):
    if not timing_enabled():
        yield
        return
    start = time.perf_counter()
    print(f"[timing] {label} ...", flush=True)
    try:
        yield
    finally:
        log_timing_msg(label, time.perf_counter() - start)


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
    timeout_seconds = float(os.environ.get("OPENROUTER_TIMEOUT", "180"))
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=concurrency + 10,
            max_keepalive_connections=concurrency,
        ),
        timeout=httpx.Timeout(timeout_seconds),
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
    with log_timing("fetch_chat_models (OpenRouter /models API)"):
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
        read_start = time.perf_counter()
        table = pq.read_table(file_path, columns=["text"])
        log_timing_msg(f"read parquet {file_path.name}", time.perf_counter() - read_start)
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


def to_pytorch_chunk(entries, indices, min_words_filter=None, generation_mode=None):
    if isinstance(indices, int):
        indices = list(range(indices, indices + len(entries)))
    chunk = {
        "texts": [entry["full_text"] for entry in entries],
        "labels": [
            torch.tensor(entry["labels"], dtype=torch.long) for entry in entries
        ],
        "models": [entry["model"] for entry in entries],
        "indices": torch.tensor(indices, dtype=torch.long),
    }
    if min_words_filter is not None:
        chunk["min_words_filter"] = int(min_words_filter)
    if generation_mode:
        chunk["generation_mode"] = generation_mode
    return chunk


def save_pytorch_chunk(
    split,
    entries,
    output_dir,
    global_indices,
    min_words_filter=None,
    generation_mode=None,
):
    if not entries:
        raise ValueError("Cannot save an empty chunk.")
    if isinstance(global_indices, int):
        global_indices = list(range(global_indices, global_indices + len(entries)))
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    global_start = min(global_indices)
    global_end = max(global_indices)
    output_path = split_dir / f"{split}_{global_start}_{global_end}.pt"
    torch.save(
        to_pytorch_chunk(
            entries,
            global_indices,
            min_words_filter=min_words_filter,
            generation_mode=generation_mode,
        ),
        output_path,
    )
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


def is_transient_error(exc):
    if isinstance(exc, (json.JSONDecodeError, APIConnectionError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, RuntimeError) and "all models unavailable" in str(exc).lower():
        return True
    return False


def is_model_content_error(exc):
    if isinstance(exc, json.JSONDecodeError):
        return False
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).lower()
    return "returned no choices" in message or "returned empty content" in message


def is_context_length_error(exc):
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
        )
    )


def is_non_retryable_entry_error(exc):
    if isinstance(exc, ValueError):
        message = str(exc)
        return (
            "words; need at least" in message
            or "Could not produce begin, middle, and end" in message
        )
    if isinstance(exc, APIStatusError) and exc.status_code == 400:
        return is_context_length_error(exc)
    return is_context_length_error(exc)


def should_retry_with_another_model(exc):
    if is_transient_error(exc):
        return True
    if is_model_content_error(exc):
        return True
    if isinstance(exc, InternalServerError):
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


class ModelBlacklist:
    """Shared run-wide registry of models to skip after failures."""

    def __init__(self):
        self._permanent = set()
        self._cooldown_until = {}
        self._rate_limit_strikes = {}
        self._lock = asyncio.Lock()

    def is_blocked(self, model, now=None):
        now = now or time.monotonic()
        if model in self._permanent:
            return True
        until = self._cooldown_until.get(model)
        return until is not None and now < until

    def available_models(self, models, now=None):
        now = now or time.monotonic()
        return [model for model in models if not self.is_blocked(model, now)]

    def seconds_until_available(self, models, now=None):
        now = now or time.monotonic()
        waits = [
            self._cooldown_until[model] - now
            for model in models
            if model in self._cooldown_until and now < self._cooldown_until[model]
        ]
        return max(0.5, min(waits)) if waits else 0.0

    def blocked_count(self, models):
        now = time.monotonic()
        return sum(1 for model in models if self.is_blocked(model, now))

    def _log_registration(self, model, reason, models):
        if not timing_enabled():
            return
        blocked = self.blocked_count(models)
        print(
            f"[blacklist] {model}: {reason} "
            f"({blocked}/{len(models)} models blocked)",
            flush=True,
        )

    async def register_failure(self, model, exc, models):
        async with self._lock:
            was_blocked = self.is_blocked(model)

            if is_daily_quota_error(exc):
                if model not in self._permanent:
                    self._permanent.add(model)
                    self._cooldown_until.pop(model, None)
                    self._log_registration(model, "daily quota exhausted", models)
                return

            if is_rate_limit_error(exc):
                strikes = self._rate_limit_strikes.get(model, 0) + 1
                self._rate_limit_strikes[model] = strikes
                if strikes >= MAX_RATE_LIMIT_STRIKES:
                    wait = min(300.0, 30.0 * strikes)
                    until = time.monotonic() + wait
                    previous = self._cooldown_until.get(model, 0)
                    self._cooldown_until[model] = max(previous, until)
                    if model in self._permanent:
                        self._permanent.discard(model)
                    if not was_blocked:
                        self._log_registration(
                            model,
                            f"rate-limited {strikes}x, cooldown {wait:.0f}s",
                            models,
                        )
                    return

                wait = parse_retry_after_seconds(exc)
                until = time.monotonic() + wait
                previous = self._cooldown_until.get(model, 0)
                self._cooldown_until[model] = max(previous, until)
                if not was_blocked:
                    self._log_registration(
                        model, f"rate-limited, cooldown {wait:.0f}s", models
                    )
                return

            if is_model_content_error(exc):
                wait = 30.0
                until = time.monotonic() + wait
                previous = self._cooldown_until.get(model, 0)
                self._cooldown_until[model] = max(previous, until)
                if not was_blocked:
                    self._log_registration(
                        model, f"empty response, cooldown {wait:.0f}s", models
                    )
                return

            if should_blacklist_model_failure(exc):
                if model not in self._permanent:
                    self._permanent.add(model)
                    self._cooldown_until.pop(model, None)
                    if not was_blocked:
                        self._log_registration(
                            model,
                            f"request failed ({exc.__class__.__name__})",
                            models,
                        )


def should_blacklist_model_failure(exc):
    if is_transient_error(exc) or is_model_content_error(exc):
        return False
    if is_rate_limit_error(exc) or is_daily_quota_error(exc):
        return False
    return should_retry_with_another_model(exc)


async def pick_model(models, blacklist):
    while True:
        async with blacklist._lock:
            available = blacklist.available_models(models)
            if available:
                return random.choice(available)
            wait = blacklist.seconds_until_available(models)
            if wait <= 0:
                raise RuntimeError(
                    f"All {len(models)} models unavailable (blacklisted or on cooldown)."
                )
        log_timing_msg("all models blocked, waiting for cooldown", wait)
        await asyncio.sleep(wait)


def extract_completion_text(response, model):
    if not response.choices:
        raise ValueError(f"Model {model} returned no choices.")
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Model {model} returned empty content.")
    return content.strip()


async def generate_text(client, prompt, words_needed, models, semaphore, model_blacklist):
    max_tokens = max(64, int(words_needed * 2) + 20)
    last_error = None
    attempts = 0
    transient_retries = 0
    max_attempts = max(MAX_MODEL_RETRIES, len(models))

    while attempts < max_attempts:
        model = await pick_model(models, model_blacklist)
        if model_blacklist.is_blocked(model):
            continue
        try:
            api_start = time.perf_counter()
            async with semaphore:
                if model_blacklist.is_blocked(model):
                    continue
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
            log_timing_msg(
                f"OpenRouter chat.completions ({model}, attempt {attempts + 1})",
                time.perf_counter() - api_start,
            )
            return extract_completion_text(response, model), model
        except Exception as exc:
            if is_transient_error(exc):
                transient_retries += 1
                if transient_retries > MAX_TRANSIENT_RETRIES:
                    raise RuntimeError(
                        f"Transient errors persisted after {MAX_TRANSIENT_RETRIES} retries."
                    ) from exc
                wait = min(60.0, TRANSIENT_RETRY_BASE_SECONDS * transient_retries)
                log_timing_msg(
                    f"transient error ({exc.__class__.__name__}), retry in",
                    wait,
                )
                await asyncio.sleep(wait)
                last_error = exc
                continue
            await model_blacklist.register_failure(model, exc, models)
            if not should_retry_with_another_model(exc) and not is_rate_limit_error(exc):
                raise
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
    min_words_filter=None,
    generation_mode=None,
):
    semaphore = asyncio.Semaphore(concurrency)
    model_blacklist = ModelBlacklist()
    total = len(entries)
    generated = 0
    skipped = 0
    failed = 0
    lock = asyncio.Lock()
    results = [None] * total
    resolved = [False] * total
    saved_chunk_ids = set()
    written_paths = []
    saved_sample_count = 0

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
        nonlocal saved_sample_count
        if split is None or output_dir is None:
            return
        for chunk_id in range((total + chunk_size - 1) // chunk_size):
            if chunk_id in saved_chunk_ids:
                continue
            start = chunk_id * chunk_size
            end = min(start + chunk_size, total)
            if not all(resolved[i] for i in range(start, end)):
                continue
            chunk_entries = []
            chunk_indices = []
            for i in range(start, end):
                if results[i] is None:
                    continue
                chunk_entries.append(results[i])
                chunk_indices.append(from_idx + i)
            saved_chunk_ids.add(chunk_id)
            if not chunk_entries:
                print(
                    f"\nChunk {from_idx + start}-{from_idx + end - 1}: all entries skipped.",
                    flush=True,
                )
                continue
            save_start = time.perf_counter()
            path = save_pytorch_chunk(
                split,
                chunk_entries,
                output_dir,
                chunk_indices,
                min_words_filter=min_words_filter,
                generation_mode=generation_mode,
            )
            log_timing_msg(f"save chunk {path.name}", time.perf_counter() - save_start)
            skipped_in_chunk = (end - start) - len(chunk_entries)
            saved_sample_count += len(chunk_entries)
            written_paths.append(path)
            for i in range(start, end):
                results[i] = None
            if skipped_in_chunk:
                print(
                    f"\nSaved {path} ({len(chunk_entries)} samples, "
                    f"{skipped_in_chunk} skipped in range)",
                    flush=True,
                )
            else:
                print(f"\nSaved {path} ({len(chunk_entries)} samples)", flush=True)

    def flush_unsaved_results():
        """Save any successful results not yet written (e.g. after partial failures)."""
        nonlocal saved_sample_count
        if split is None or output_dir is None:
            return
        chunk_entries = []
        chunk_indices = []
        for i in range(total):
            if results[i] is not None:
                chunk_entries.append(results[i])
                chunk_indices.append(from_idx + i)
        if not chunk_entries:
            return
        save_start = time.perf_counter()
        path = save_pytorch_chunk(
            split,
            chunk_entries,
            output_dir,
            chunk_indices,
            min_words_filter=min_words_filter,
            generation_mode=generation_mode,
        )
        log_timing_msg(f"save remaining {path.name}", time.perf_counter() - save_start)
        saved_sample_count += len(chunk_entries)
        written_paths.append(path)
        print(
            f"\nSaved remaining results to {path} ({len(chunk_entries)} samples)",
            flush=True,
        )
        for i in range(total):
            results[i] = None

    async def run_one(index, entry):
        nonlocal generated, skipped, failed
        last_error = None
        global_index = from_idx + index

        for entry_attempt in range(MAX_ENTRY_RETRIES):
            try:
                entry_start = time.perf_counter()
                result = await create_one(
                    client, entry, models, semaphore, model_blacklist
                )
                if not is_valid_result(result):
                    raise ValueError(
                        f"Entry {global_index} did not produce valid generated text."
                    )
                async with lock:
                    results[index] = result
                    resolved[index] = True
                    generated += 1
                    print_progress()
                    save_ready_chunks()
                log_timing_msg(
                    f"entry {global_index} total"
                    + (
                        f" (retry {entry_attempt + 1})"
                        if entry_attempt > 0
                        else ""
                    ),
                    time.perf_counter() - entry_start,
                )
                return
            except Exception as exc:
                last_error = exc
                if is_non_retryable_entry_error(exc):
                    print(f"\nSkipping entry {global_index}: {exc}", flush=True)
                    async with lock:
                        resolved[index] = True
                        skipped += 1
                        save_ready_chunks()
                    return
                if entry_attempt + 1 >= MAX_ENTRY_RETRIES:
                    break
                if is_daily_quota_error(exc):
                    break
                if is_transient_error(exc) or "all models unavailable" in str(
                    exc
                ).lower():
                    wait = max(
                        model_blacklist.seconds_until_available(models),
                        TRANSIENT_RETRY_BASE_SECONDS * (entry_attempt + 1),
                    )
                elif is_temporary_rate_limit(exc):
                    wait = parse_retry_after_seconds(exc)
                else:
                    wait = 5 * (entry_attempt + 1)
                log_timing_msg(
                    f"entry {global_index} backoff before retry",
                    wait,
                )
                print(
                    f"\nEntry {global_index} failed "
                    f"({entry_attempt + 1}/{MAX_ENTRY_RETRIES}), "
                    f"retrying in {wait:.0f}s: {exc.__class__.__name__}",
                    flush=True,
                )
                await asyncio.sleep(wait)

        print(
            f"\nSkipping entry {global_index} after {MAX_ENTRY_RETRIES} attempts: {last_error}",
            flush=True,
        )
        async with lock:
            resolved[index] = True
            failed += 1
            save_ready_chunks()
        return

    try:
        with log_timing(f"run_sequences ({total} entries)"):
            await asyncio.gather(*(run_one(i, entry) for i, entry in enumerate(entries)))
    finally:
        save_ready_chunks()
        flush_unsaved_results()
    print()
    print(
        f"Done: {generated}/{total} generated, {skipped} skipped, "
        f"{failed} failed, {saved_sample_count} saved to disk.",
        flush=True,
    )
    if written_paths:
        for path in written_paths:
            print(f"  {path}", flush=True)
    elif split and output_dir and generated:
        print(
            f"  Warning: {generated} samples generated but no chunk file was written.",
            flush=True,
        )
    if skipped:
        print(
            f"Skipped {skipped}/{total} entries (too short or unsplittable).",
            flush=True,
        )
    if failed:
        print(
            f"Failed {failed}/{total} entries (API errors after retries).",
            flush=True,
        )
    return saved_sample_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate human/AI mixed datasets from MiniPile parquet splits."
    )
    parser.add_argument(
        "--mode",
        choices=["append", "mixed_v2", "sandwitch", "sandwitch_v2"],
        default=os.environ.get("GENERATION_MODE", "sandwitch"),
        help=(
            "append: random-cutoff human+AI; mixed_v2: slice, summarize suffix, regenerate continuation; "
            "sandwitch: human+AI+human; sandwitch_v2: summarize middle then regenerate."
        ),
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
