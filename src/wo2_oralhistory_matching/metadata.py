import json
from .prompts import _build_extract_name_prompt, _build_segment_title_prompt
from .batching import _first_batch_with_x_minutes_of_captions
from .models import Caption
from .chat_router import _chat
from .response_cleaner import _clean_json_output

__all__ = [
    "extract_name_from_transcript",
    "add_metadata_to_enriched_segment",
]

def extract_name_from_transcript(captions: list[Caption], api_key: str, model: str = 'gpt-4.1') -> str:
    batch = _first_batch_with_x_minutes_of_captions(captions = captions, max_minutes = 5)    
    prompt = _build_extract_name_prompt(batch)
    response_text = _chat(prompt, api_key=api_key, model=model, system_message = "You help to extract metadata from interview transcription.")
    clean_response = _clean_json_output(response_text)

    try:
        name_json = json.loads(clean_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return None

    if not name_json:
        return None

    if isinstance(name_json, dict):
        return name_json.get("name")

    if isinstance(name_json, list):
        for item in name_json:
            if isinstance(item, dict) and "name" in item:
                return item["name"]
    return None

def add_metadata_to_enriched_segment(serialized_segment_json, api_key: str, model: str = 'gpt-4.1'):
    enriched_with_metadata = []

    for segment in serialized_segment_json:
        prompt = _build_segment_title_prompt(segment)

        response_text = _chat(prompt, api_key=api_key, model=model, system_message="You create short titles for interview segments")

        clean_response = _clean_json_output(response_text)

        title = None
        try:
            parsed = json.loads(clean_response)
            if isinstance(parsed, dict):
                title = parsed.get("title")
        except json.JSONDecodeError:
            pass

        segment_with_metadata = {"segment_title": title, **segment}
        enriched_with_metadata.append(segment_with_metadata)

    return enriched_with_metadata