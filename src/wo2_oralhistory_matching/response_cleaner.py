def _clean_json_output(response_text: str) -> str:
    """
    Remove any Markdown highlighting.
    """
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    elif response_text.startswith("<think>"):
        response_text = response_text.split('</think>')[1]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text.strip()