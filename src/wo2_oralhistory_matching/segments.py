import json
from .models import Caption, Segment
from .prompts import _build_segment_prompt, _build_segment_selector_prompt
from .chat_router import _chat
from .response_cleaner import _clean_json_output
from .batching import _first_batch_with_x_minutes_of_captions, _batch_segments_by_tokens

__all__ = [
    "create_segments_from_captions",
    "create_segments_from_boundaries",
    "select_segments_to_be_enriched"
]

def _build_segment_from_indices(captions: list[Caption], indices: list[int]) -> Segment:
    """
    Constructs a Segment object from caption indices and caption list.
    """
    segment_captions = [captions[idx] for idx in indices if 0 <= idx < len(captions)]

    if not segment_captions:
        raise ValueError("Empty caption list for segment")

    start = segment_captions[0].start
    end = segment_captions[-1].end
    text = " ".join(c.text.replace("\n", " ").strip() for c in segment_captions)

    return Segment(start=start, end=end, text=text, captions=segment_captions)

def _segment_with_llm(captions: list[Caption], api_key: str, model: str = "gpt-4.1", minutes_per_batch: int = 20) -> list[Segment]:
    """
    Creates segments in batches, by asking the LLM to create segments.
    """
    all_segments = []
    
    next_caption_index = 0
    stuck_counter = 0

    while next_caption_index < len(captions):
        remaining_captions = captions[next_caption_index:]
        batch_captions = _first_batch_with_x_minutes_of_captions(remaining_captions, minutes_per_batch)
        variation_suffix = ""

        if stuck_counter > 0:
            variation_suffix = f"# This is retry number {stuck_counter}. Please make sure to create segments based on the rules above."

        prompt = _build_segment_prompt(batch_captions, index_offset=next_caption_index, variation_suffix=variation_suffix)

        response_text = _chat(prompt, api_key=api_key, model=model, system_message = "You help to split interview transcripts into segments.")
        clean_response = _clean_json_output(response_text)

        try:
            segments_data = json.loads(clean_response)
        except json.JSONDecodeError as e:
            break
        
        if not segments_data:
            break

        last_segment = None
        last_caption_indices = None

        for i, seg_dict in enumerate(segments_data):
            caption_indices = seg_dict["caption_indices"]

            try:
                segment = _build_segment_from_indices(captions, caption_indices)
            except ValueError:
                continue

            if i == len(segments_data) - 1:
                last_segment = segment
                last_caption_indices = caption_indices
            else:
                all_segments.append(segment)
        
        if last_caption_indices and last_caption_indices[-1] >= len(captions) - 1:
            all_segments.append(last_segment)
            break

        if last_caption_indices:
            last_start = last_caption_indices[0]

            if last_start <= next_caption_index:
                stuck_counter += 1
                if stuck_counter < 3:
                    continue
                else:
                    next_caption_index = next_caption_index + len(batch_captions)
                    stuck_counter = 0
                    continue
            else:
                stuck_counter = 0
                next_caption_index = last_start
        else:
            break
    return all_segments


def create_segments_from_captions(captions: list[Caption], api_key: str, model: str = "gpt-4.1", minutes_per_batch: int = 20) -> list[Segment]:
    """
    Processes a list of captions and performs segmentation.
    """
    segments = _segment_with_llm(captions, api_key, model, minutes_per_batch)
    return segments

def create_segments_from_boundaries(captions: list[Caption], boundaries: list[int]) -> list[Segment]:
    """
    Creates segments based on manually set boundaries.
    """
    segments = []
    for i, start_idx in enumerate(boundaries):
        end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(captions)
        indices = list(range(start_idx, end_idx))

        segment = _build_segment_from_indices(captions, indices)
        segments.append(segment)

    return segments

def select_segments_to_be_enriched(segments: list[Segment], api_key: str, model: str = "gpt-4.1", max_tokens: int = 800000) -> list[Segment]:
    selected_segments = []
    batches = _batch_segments_by_tokens(segments, model=model, max_tokens= max_tokens)
    for batch in batches:
        prompt = _build_segment_selector_prompt(batch)
        response_text = _chat(prompt, api_key=api_key, model=model, system_message = "You help determine which interview segments contain valuable content about WW2.")
        try:
            clean_response = _clean_json_output(response_text)
            if isinstance(clean_response, str):
                clean_response = json.loads(clean_response)

            if isinstance(clean_response, dict):
                selected_indices = clean_response.get("relevant_segments", [])
            elif isinstance(clean_response, list):
                selected_indices = clean_response
            else:
                selected_indices = []
            for i in selected_indices:
                if 0 <= i < len(batch):
                    selected_segments.append(batch[i])
        except Exception as e:
            print("Fout bij het parsen van LLM-output", e)
            print("Antwoord was:", response_text)
    return selected_segments