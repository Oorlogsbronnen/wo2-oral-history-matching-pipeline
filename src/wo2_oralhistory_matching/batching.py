import tiktoken
from .prompts import _build_segment_selector_prompt, _build_topdown_matching_prompt
from .models import Segment, Caption

def _get_encoding_for_model(model:str = "gpt-4.1"):
    """
    Get the encoding for an OpenAI model using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    return encoding

def _first_batch_with_x_minutes_of_captions(captions: list[Caption], max_minutes: int = 20) -> list[Caption]:
    """
    Returns the first batch of captions covering up to `max_duration` seconds,
    based on the `end` time of each caption.
    """
    if not captions:
        return []
    
    batch = []
    max_duration = max_minutes * 60
    end_limit = captions[0].end + max_duration

    for caption in captions:
        if caption.end <= end_limit:
            batch.append(caption)
        else:
            break

    return batch

def _batch_segments_by_tokens(segments: list[Segment], model: str = "gpt-4.1", max_tokens: int = "800000") -> list[list[Segment]]:
    """
    Splits the segments into batches based on the maximum amount of tokens allowed per request.
    """
    encoding = _get_encoding_for_model(model)

    dummy_prompt = _build_segment_selector_prompt([])
    dummy_prompt_tokens = len(encoding.encode(dummy_prompt))

    batches = []
    current_batch = []
    current_tokens = 0

    for segment in segments:
        text = segment.text.replace("\n", " ").strip()
        token_count = len(encoding.encode(text)) + 10

        if current_tokens + token_count + dummy_prompt_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        current_batch.append(segment)
        current_tokens += token_count

    if current_batch:
        batches.append(current_batch)
    return batches

def _batch_concept_labels_by_tokens(labels: list[str], segment_text: str, model: str = "gpt-4.1", max_tokens: int = "800000") -> list[list[str]]:
    """
    Splits the concepts into batches based on the maximum amount of tokens allowed per request.
    """
    encoding = _get_encoding_for_model(model)

    dummy_prompt = _build_topdown_matching_prompt(concept_labels=[], segment_text=segment_text)
    dummy_prompt_tokens = len(encoding.encode(dummy_prompt))

    available_tokens = max(0, max_tokens - dummy_prompt_tokens)

    batches = []
    current_batch = []
    current_tokens = 0

    for label in labels:
        label_token_count = len(encoding.encode(label)) + 5

        if current_tokens + label_token_count > available_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(label)
        current_tokens += label_token_count

    if current_batch:
        batches.append(current_batch)

    return batches