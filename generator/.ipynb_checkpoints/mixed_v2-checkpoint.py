import os
import random

from sandwitch_v2 import get_sentences
from utils import generate_text, run_sequences

MIN_MIXED_V2_WORDS = 30
MIN_MIXED_V2_SENTENCES = 2
MIN_PART_WORDS = 2

generation_prompts = [
    '''You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. 
    You will see the beginning and ending of a text. Regenerate ending text that fits seamlessly with the beginning section. 
    Retain any disclaimers by rephrasing them, and avoid providing any additional suffix. 
    Do not generate anything else (Only regenerated ending part that can be continued smoothly by the beginning section).'''
]


def split_sentences_into_prefix_suffix(sentences):
    """Split at ~1/3 of sentences (by count), balancing character mass slightly."""
    lens = [len(part) for part in sentences]
    split_at = max(1, len(sentences) // 2)

    prefix_size = sum(lens[:split_at])
    suffix_size = sum(lens[split_at:])

    for _ in range(10):
        if split_at <= 1 or split_at >= len(sentences) - 1:
            break
        if prefix_size > suffix_size and split_at > 1:
            split_at -= 1
            prefix_size = sum(lens[:split_at])
            suffix_size = sum(lens[split_at:])
        elif (
            prefix_size + lens[split_at] < suffix_size - lens[split_at]
            and split_at < len(sentences) - 1
        ):
            split_at += 1
            prefix_size = sum(lens[:split_at])
            suffix_size = sum(lens[split_at:])
        else:
            break

    prefix = "".join(sentences[:split_at])
    suffix = "".join(sentences[split_at:])
    return prefix, suffix


def split_words_into_prefix_suffix(text):
    words = text.split()
    if len(words) < MIN_MIXED_V2_WORDS:
        raise ValueError(
            f"Text has {len(words)} words; need at least {MIN_MIXED_V2_WORDS}."
        )

    prefix_len = max(MIN_PART_WORDS, len(words) // 2)
    if len(words) - prefix_len < MIN_PART_WORDS:
        raise ValueError(
            "Could not produce prefix and suffix with enough words in each part."
        )

    prefix = " ".join(words[:prefix_len])
    suffix = " ".join(words[prefix_len:])
    return prefix, suffix


def parts_have_enough_words(prefix, suffix):
    return (
        len(prefix.split()) >= MIN_PART_WORDS
        and len(suffix.split()) >= MIN_PART_WORDS
    )


def split_into_prefix_and_suffix(text):
    """Human prefix (kept) + ending suffix (summarized, then used to guide full-text generation)."""
    sentences = get_sentences(text)
    if len(sentences) >= MIN_MIXED_V2_SENTENCES:
        prefix, suffix = split_sentences_into_prefix_suffix(sentences)
        if parts_have_enough_words(prefix, suffix):
            return prefix, suffix
    return split_words_into_prefix_suffix(text)


def build_summary_prompt(ending_text, summary_prompt):
    return f"{summary_prompt}\n\nText:\n{ending_text}"


def build_full_text_generation_prompt(
    prefix, suffix, generation_prompt
):
    return (
        f"{generation_prompt}\n\n"
        f"The generated suffix portion should be about {len(suffix.split())} words long.\n\n"
        f"beginning section: {prefix}\n"
        f"ending section: {suffix}"
    )


def build_mixed_v2_result(prefix, regenerated_suffix, generation_model):
    prefix_words = prefix.split()
    ai_words = regenerated_suffix.strip().split()
    full_text = " ".join(prefix_words + ai_words)

    labels = [0] * len(prefix_words) + [1] * len(ai_words)
    return {
        "full_text": full_text,
        "labels": labels,
        "model": f"{generation_model}",
    }


async def create_mixed_v2_sequence(
    client, human_text, models, semaphore, model_blacklist
):
    prefix, suffix = split_into_prefix_and_suffix(human_text)
    full_text_word_count = len(prefix.split()) + len(suffix.split())
    generation_prompt = random.choice(generation_prompts)

    regenerated_suffix, generation_model = await generate_text(
        client,
        build_full_text_generation_prompt(
            prefix, suffix, generation_prompt
        ),
        full_text_word_count,
        models,
        semaphore,
        model_blacklist,
    )

    return build_mixed_v2_result(
        prefix, regenerated_suffix, generation_model
    )


async def create_mixed_v2_sequences(
    client, human_entries, models, concurrency, **save_kwargs
):
    return await run_sequences(
        create_mixed_v2_sequence,
        client,
        human_entries,
        models,
        concurrency,
        min_words_filter=MIN_MIXED_V2_WORDS,
        generation_mode="mixed_v2",
        **save_kwargs,
    )
