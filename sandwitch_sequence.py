import os
import random

from utils import generate_text, run_sequences

MIN_SANDWITCH_WORDS = 15


def prepare_sandwitch_sequence(human_text):
    """Slice human text into start + end segments with a gap for AI bridge text."""
    ai_words_min = int(os.environ.get("OPENROUTER_AI_WORDS_MIN", "5"))
    ai_words_max = int(os.environ.get("OPENROUTER_AI_WORDS_MAX", "100"))
    human_words = human_text.split()
    total_human_len = len(human_words)
    words_needed = random.randint(ai_words_min, ai_words_max)

    max_start = min(10, total_human_len // 3)
    start_cutoff = random.randint(3, max(3, max_start))
    remaining = total_human_len - start_cutoff
    max_end = min(10, remaining - 1)
    end_cutoff = random.randint(3, max(3, max_end))

    human_start_tokens = human_words[:start_cutoff]
    human_end_tokens = human_words[-end_cutoff:]
    human_start_text = " ".join(human_start_tokens)
    human_end_text = " ".join(human_end_tokens)

    prompt = (
        f"You are a text generation bridge. Read the START text and the END text below. "
        f"Write exactly {words_needed} words that seamlessly connect the START directly to the END. "
        f"Do not repeat the prompt, the start text, or the end text. Output ONLY your bridge text.\n\n"
        f"--- START ---\n{human_start_text}\n"
        f"--- END ---\n{human_end_text}"
    )

    return human_start_tokens, human_end_tokens, prompt, words_needed


def build_sandwitch_result(human_start_tokens, human_end_tokens, ai_text, model):
    ai_words = ai_text.split()
    all_words = human_start_tokens + ai_words + human_end_tokens
    labels = (
        [0] * len(human_start_tokens)
        + [1] * len(ai_words)
        + [0] * len(human_end_tokens)
    )

    return {
        "full_text": " ".join(all_words),
        "labels": labels,
        "model": model,
    }


async def create_sandwitch_sequence(client, human_text, models, semaphore, failed_models):
    human_start, human_end, prompt, words_needed = prepare_sandwitch_sequence(human_text)
    ai_text, model = await generate_text(
        client, prompt, words_needed, models, semaphore, failed_models
    )
    return build_sandwitch_result(human_start, human_end, ai_text, model)


async def create_sandwitch_sequences(client, human_entries, models, concurrency, **save_kwargs):
    return await run_sequences(
        create_sandwitch_sequence,
        client,
        human_entries,
        models,
        concurrency,
        **save_kwargs,
    )
