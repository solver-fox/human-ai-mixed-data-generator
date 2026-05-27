import os
import random
import re

from utils import generate_text, run_sequences

MIN_SANDWITCH_V2_WORDS = 30
MIN_SANDWITCH_V2_SENTENCES = 3
MIN_PART_WORDS = 2

summary_prompts = [
    "Summarize the text in your own words, highlighting the key points. Do not generate anything else.",
    "Provide a concise summary of the text, focusing on its main argument. Refrain from generating anything else.",
    "In a few sentences, capture the core ideas of the text. Ensure you do not produce anything else.",
    "Write a short overview of the text, emphasizing the primary takeaways. Do not include anything else beyond the summary.",
    "Condense the text into a brief summary, touching on the essential details. Do not provide anything else in your response.",
    "Explain the text's main points in a summarized format. Nothing else should be generated.",
    "Give me a succinct summary of the text's content. Do not produce additional information.",
    "What is the most important information to include in a summary of this text? Only produce the summary, nothing else.",
    "Craft a concise review of the text, highlighting the central message. No other content should be added.",
    "Generate a quick summary that identifies the text's key themes. Provide only the summary, with nothing else included.",
    "Offer a short synopsis of the text, noting the critical arguments. Please do not add anything else.",
    "Provide an executive summary of the text's main findings. Avoid including extra information.",
    "Distill the text into a paragraph covering the core ideas. Refrain from adding any additional content.",
    "Summarize the text with an emphasis on its conclusion and supporting points. Do not provide anything beyond the summary.",
    "In just a few sentences, outline the text's primary purpose and insights. Do not generate anything else.",
]

generation_prompts = [
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will be given the start and finish of a text plus a summary of its middle. Your job is to compose only the middle portion, making sure it aligns with both the beginning and the end. Do not provide a summary; preserve any existing warnings by rephrasing them, and write nothing else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You receive the opening and closing paragraphs of a text, as well as a synopsis of the central section. Your task is to generate the text for the middle part alone, ensuring coherence with the given beginning and end. Keep any cautions or alerts by rewording them, and do not include any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You are provided with a text's first and final segments along with a brief outline of what occurs in the middle. Your job is to fill in only the middle content. The final text should flow naturally, so do not insert a summary. Retain all warnings by rephrasing, and write nothing else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the initial and concluding parts of a text, plus a summary that describes the middle portion. Construct only the middle section so that it fits seamlessly from start to end. Rephrase and keep any warnings, and do not add a summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will see the beginning and ending of a text and a concise description of its midpoint. Your role is to write only the middle paragraphs, ensuring coherence with the provided segments. Maintain any disclaimers by rephrasing them, and avoid including any summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You get the start and finish of a text, as well as a summary of what happens in between. Create only the central portion, ensuring logical flow without adding a recap. Retain all cautions by rephrasing them as needed, and do not write anything else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. The opening and concluding lines of a text, plus a synopsis of the middle, will be given to you. Your aim is to produce only the middle content. Preserve existing warnings in a rephrased form, and refrain from including any summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the introduction and conclusion of a text, together with an overview of the midsection. Craft only the missing middle text so that the entire piece remains coherent. Keep any alerts or disclaimers by rewording them, and omit any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. Given the first and last parts of a text, plus a short summary of the middle, your job is to write only the central portion. Maintain coherence with the given sections, preserve any warnings by rephrasing them, and do not summarize anything. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will be shown the beginning and ending of a text along with a high-level summary of its midpoint. Only generate the middle content to ensure a continuous flow. Any existing notices or cautions must be included but reworded, and avoid all summarization. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You receive the text's start and end segments, plus a brief overview of the middle. Construct only the middle text, ensuring it aligns with the summary and merges naturally with the given parts. Rephrase and keep warnings intact, without including a summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You are given the initial and final parts of a text, along with a concise account of what takes place in the middle. Write only the middle paragraphs to form a cohesive piece. Adjust any existing warnings but keep them, and exclude all summaries. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. The beginning and ending sections of a text, along with an abstract of the middle, are provided. Your task: generate only the missing central portion so the entire text reads coherently. Reword any cautions, and do not include any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the introduction and final segment of a text, plus a summary of the events in between. Craft the middle portion only, preserving flow. Keep warnings by restating them in your own words, and do not add any form of summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will see the first and last paragraphs of a text plus a synopsis of the middle. Provide only the middle text, ensuring it fits seamlessly. Retain any disclaimers by rephrasing them, and avoid providing any additional summarization. Do not generate anything else (Only middle part - you're output will be concatenated with begin and end).",
]


def get_sentences(text):
    text = text.strip()
    if not text:
        return []
    sentences = re.findall(r"[^.!?]*[.!?]+|[^.!?]+$", text, flags=re.DOTALL)
    sentences = [part for part in sentences if part.strip()]
    return sentences if sentences else [text]


def split_sentences_into_three_parts(sentences):
    """Friend's sentence split with balanced character sizes."""
    lens = [len(part) for part in sentences]
    first_part = len(sentences) // 3
    second_part = 2 * len(sentences) // 3

    first_size = sum(lens[:first_part])
    second_size = sum(lens[first_part:second_part])
    third_size = sum(lens[second_part:])

    for _ in range(10):
        if first_part <= 1 or second_part <= first_part + 1 or second_part >= len(sentences) - 1:
            break
        if first_size - lens[first_part - 1] > second_size + lens[first_part - 1]:
            first_part -= 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_size - lens[second_part - 1] > third_size + lens[second_part - 1]:
            second_part -= 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        elif first_size + lens[first_part] < second_size - lens[first_part]:
            first_part += 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_size + lens[second_part] < third_size - lens[second_part]:
            second_part += 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        else:
            break

    begin = "".join(sentences[:first_part])
    middle = "".join(sentences[first_part:second_part])
    end = "".join(sentences[second_part:])

    middle_stripped = middle.rstrip()
    trailing_whitespace = middle[len(middle_stripped) :]
    end = trailing_whitespace + end
    middle = middle_stripped

    return begin, middle, end


def parts_have_enough_words(begin, middle, end):
    return all(len(part.split()) >= MIN_PART_WORDS for part in (begin, middle, end))


def split_words_into_three_parts(text):
    """Fallback when there are too few sentences: split by word count in thirds."""
    words = text.split()
    if len(words) < MIN_SANDWITCH_V2_WORDS:
        raise ValueError(
            f"Text has {len(words)} words; need at least {MIN_SANDWITCH_V2_WORDS}."
        )

    total = len(words)
    first_part = max(MIN_PART_WORDS, total // 3)
    second_part = max(first_part + MIN_PART_WORDS, (2 * total) // 3)
    if second_part > total - MIN_PART_WORDS:
        second_part = total - MIN_PART_WORDS
    if first_part >= second_part:
        first_part = MIN_PART_WORDS
        second_part = total - MIN_PART_WORDS
    if second_part - first_part < MIN_PART_WORDS or total - second_part < MIN_PART_WORDS:
        raise ValueError("Could not produce begin, middle, and end with enough words.")

    begin = " ".join(words[:first_part])
    middle = " ".join(words[first_part:second_part])
    end = " ".join(words[second_part:])
    return begin, middle, end


def split_into_three_parts(text):
    """Split text into begin / middle / end; sentences preferred, words as fallback."""
    sentences = get_sentences(text)
    if len(sentences) >= MIN_SANDWITCH_V2_SENTENCES:
        begin, middle, end = split_sentences_into_three_parts(sentences)
        if parts_have_enough_words(begin, middle, end):
            return begin, middle, end
    return split_words_into_three_parts(text)


def build_summary_prompt(middle_text, summary_prompt):
    return f"{summary_prompt}\n\nText:\n{middle_text}"


def build_generation_prompt(begin, end, summary, generation_prompt, middle_word_count):
    return (
        f"{generation_prompt} The middle should be about {middle_word_count} words long.\n\n"
        f"begin: {begin}\n"
        f"end: {end}\n"
        f"summary: {summary}"
    )


def build_sandwitch_v2_result(begin, end, generated_middle, summary_model, generation_model):
    begin_words = begin.split()
    middle_words = generated_middle.strip().split()
    end_words = end.split()
    all_words = begin_words + middle_words + end_words
    full_text = " ".join(all_words)
    labels = [0] * len(begin_words) + [1] * len(middle_words) + [0] * len(end_words)

    return {
        "full_text": full_text,
        "labels": labels,
        "model": f"{summary_model}+{generation_model}",
    }


async def create_sandwitch_v2_sequence(
    client, human_text, models, semaphore, model_blacklist
):
    begin, middle, end = split_into_three_parts(human_text)
    middle_word_count = len(middle.split())
    summary_words = max(20, min(middle_word_count // 2, 150))

    summary_prompt = random.choice(summary_prompts)
    generation_prompt = random.choice(generation_prompts)

    summary, summary_model = await generate_text(
        client,
        build_summary_prompt(middle, summary_prompt),
        summary_words,
        models,
        semaphore,
        model_blacklist,
    )

    generated_middle, generation_model = await generate_text(
        client,
        build_generation_prompt(begin, end, summary, generation_prompt, middle_word_count),
        middle_word_count,
        models,
        semaphore,
        model_blacklist,
    )

    return build_sandwitch_v2_result(
        begin, end, generated_middle, summary_model, generation_model
    )


async def create_sandwitch_v2_sequences(
    client, human_entries, models, concurrency, **save_kwargs
):
    return await run_sequences(
        create_sandwitch_v2_sequence,
        client,
        human_entries,
        models,
        concurrency,
        **save_kwargs,
    )
