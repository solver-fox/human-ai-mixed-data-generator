import asyncio
import json
import os
import random
from openai import AsyncOpenAI

CONCURRENCY = 10


def _get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY (recommended) or OPENAI_API_KEY."
        )

    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _prepare_sequence(human_text):
    human_words = human_text.split()
    word_cutoff = random.randint(3, min(100, len(human_words)))
    human_subtext = " ".join(human_words[:word_cutoff])
    prompt = (
        "Continue writing the following text naturally without repeating the prompt:\n\n"
        f"{human_subtext}"
    )
    return human_words, word_cutoff, prompt


def _build_result(human_words, word_cutoff, ai_append_text):
    ai_words = ai_append_text.split()
    all_words = human_words[:word_cutoff] + ai_words
    labels = [0] * word_cutoff + [1] * len(ai_words)

    return {
        "full_text": " ".join(all_words),
        "labels": labels,  # 0 for human, 1 for AI
    }


async def create_mixed_sequence(client, human_text, semaphore):
    human_words, word_cutoff, prompt = _prepare_sequence(human_text)

    async with semaphore:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

    ai_append_text = response.choices[0].message.content.strip()
    return _build_result(human_words, word_cutoff, ai_append_text)


async def create_mixed_sequences(client, human_entries, concurrency=CONCURRENCY):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        create_mixed_sequence(client, entry, semaphore)
        for entry in human_entries
    ]
    return await asyncio.gather(*tasks)


async def main():
    input_path = "human_dataset_mini.json"
    output_path = "mixed_dataset.json"

    with open(input_path, encoding="utf-8") as f:
        human_entries = json.load(f)

    client = _get_client()
    mixed_entries = await create_mixed_sequences(client, human_entries)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mixed_entries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
