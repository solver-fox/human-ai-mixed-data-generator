import asyncio
import json
import os
import random
import urllib.request
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, NotFoundError, RateLimitError

load_dotenv()

CONCURRENCY = 10
MAX_MODEL_RETRIES = 10
MODELS_API_URL = "https://openrouter.ai/api/v1/models?output_modalities=text"


def _get_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY in your environment or .env file."
        )
    return api_key


def _get_client(api_key):
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _fetch_chat_models(api_key):
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
        if model.get("expiration_date"):
            continue
        models.append(model_id)

    if not models:
        raise RuntimeError("No chat-capable text models returned from OpenRouter.")

    return models


def _get_models(api_key):
    env_models = os.environ.get("OPENROUTER_MODELS")
    if env_models:
        return [model.strip() for model in env_models.split(",") if model.strip()]
    return _fetch_chat_models(api_key)


def _prepare_sequence(human_text):
    human_words = human_text.split()
    word_cutoff = random.randint(3, min(100, len(human_words)))
    human_subtext = " ".join(human_words[:word_cutoff])
    prompt = (
        "Continue writing the following text naturally without repeating the prompt:\n\n"
        f"{human_subtext}"
    )
    return human_words, word_cutoff, prompt


def _should_retry_with_another_model(exc):
    if isinstance(exc, (NotFoundError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code in (403, 404, 502, 503):
            return True
        if exc.status_code == 400:
            message = str(exc).lower()
            return "model" in message and (
                "not found" in message
                or "does not exist" in message
                or "do not have access" in message
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


async def create_mixed_sequence(client, human_text, models, semaphore, failed_models):
    human_words, word_cutoff, prompt = _prepare_sequence(human_text)
    last_error = None

    for _ in range(MAX_MODEL_RETRIES):
        model = _pick_model(models, failed_models)
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
            ai_append_text = response.choices[0].message.content.strip()
            return _build_result(human_words, word_cutoff, ai_append_text, model)
        except Exception as exc:
            if not _should_retry_with_another_model(exc):
                raise
            failed_models.add(model)
            last_error = exc

    raise RuntimeError(
        f"Failed after {MAX_MODEL_RETRIES} model attempts (last model unavailable)."
    ) from last_error


async def create_mixed_sequences(client, human_entries, models, concurrency=CONCURRENCY):
    semaphore = asyncio.Semaphore(concurrency)
    failed_models = set()
    tasks = [
        create_mixed_sequence(client, entry, models, semaphore, failed_models)
        for entry in human_entries
    ]
    return await asyncio.gather(*tasks)


async def main():
    input_path = "human_dataset_mini.json"
    output_path = "mixed_dataset_mini.json"

    with open(input_path, encoding="utf-8") as f:
        human_entries = json.load(f)

    api_key = _get_api_key()
    client = _get_client(api_key)
    models = _get_models(api_key)
    print(f"Using {len(models)} OpenRouter chat models (random per entry).")
    mixed_entries = await create_mixed_sequences(client, human_entries, models)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mixed_entries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
