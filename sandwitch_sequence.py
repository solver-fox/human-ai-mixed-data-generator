import random

def _prepare_sandwich_sequence(human_text, target_total=35):
    """
    Slices human text to leave a gap in the middle for the AI to fill.
    Maintains a final target word count.
    """
    human_words = human_text.split()
    total_human_len = len(human_words)
    
    # 1. Enforce length boundaries to guarantee room for AI and trailing human text
    # Human start (3-10 words), Human end (3-10 words)
    start_cutoff = random.randint(3, min(10, total_human_len // 3))
    end_cutoff = random.randint(3, min(10, total_human_len // 3))
    
    human_start_tokens = human_words[:start_cutoff]
    # Grab tokens from the end of the original text
    human_end_tokens = human_words[-end_cutoff:]
    
    human_start_text = " ".join(human_start_tokens)
    human_end_text = " ".join(human_end_tokens)
    
    # 2. Calculate remaining word budget for the AI
    current_human_count = len(human_start_tokens) + len(human_end_tokens)
    words_needed_from_ai = max(5, target_total - current_human_count)
    
    # 3. Prompt instructing the LLM to bridge the specific gap
    prompt = (
        f"You are a text generation bridge. Read the START text and the END text below. "
        f"Write exactly {words_needed_from_ai} words that seamlessly connect the START directly to the END. "
        f"Do not repeat the prompt, the start text, or the end text. Output ONLY your bridge text.\n\n"
        f"--- START ---\n{human_start_text}\n"
        f"--- END ---\n{human_end_text}"
    )
    
    # Return structures needed for the processing step
    return human_start_tokens, human_end_tokens, prompt


def format_to_target_schema(human_start, ai_middle, human_end, model_name="x-ai/grok-4.20"):
    """
    Combines text segments and outputs the exact requested JSON format.
    """
    start_words = human_start.strip().split()
    middle_words = ai_middle.strip().split()
    end_words = human_end.strip().split()
    
    full_text = " ".join(start_words + middle_words + end_words)
    labels = * len(start_words) + * len(middle_words) + * len(end_words)
    
    return {
        "full_text": full_text,
        "labels": labels,
        "model": model_name
    }
