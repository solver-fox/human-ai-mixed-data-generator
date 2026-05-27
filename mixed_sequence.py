import os
import random

from utils import generate_text, run_sequences


def prepare_sequence(human_text):
    ai_words_min = int(os.environ.get("OPENROUTER_AI_WORDS_MIN", "10"))
    ai_words_max = int(os.environ.get("OPENROUTER_AI_WORDS_MAX", "100"))

    human_words = human_text.split()
    word_cutoff = random.randint(3, min(100, len(human_words)))
    human_subtext = " ".join(human_words[:word_cutoff])
    words_needed = random.randint(ai_words_min, ai_words_max)
    prompt = (
        f"Continue writing the following text naturally. Generate approximately {words_needed} words, "
        f"so that the final combined text reaches a grand total of {words_needed + word_cutoff} words. "
        f"Do not repeat the prompt or the input text:\n\n{human_subtext}"
    )
    return human_words, word_cutoff, prompt, words_needed


def build_result(human_words, word_cutoff, ai_text, model):
    ai_words = ai_text.split()
    all_words = human_words[:word_cutoff] + ai_words
    labels = [0] * word_cutoff + [1] * len(ai_words)

    return {
        "full_text": " ".join(all_words),
        "labels": labels,
        "model": model,
    }


async def create_mixed_sequence(client, human_text, models, semaphore, model_blacklist):
    human_words, word_cutoff, prompt, words_needed = prepare_sequence(human_text)
    ai_text, model = await generate_text(
        client, prompt, words_needed, models, semaphore, model_blacklist
    )
    return build_result(human_words, word_cutoff, ai_text, model)


async def create_mixed_sequences(client, human_entries, models, concurrency, **save_kwargs):
    return await run_sequences(
        create_mixed_sequence,
        client,
        human_entries,
        models,
        concurrency,
        **save_kwargs,
    )
