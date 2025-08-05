from .openai_client import _safe_chat_call as openai_chat

def _chat(prompt, api_key: str, model: str, system_message: str = "") -> str:
    """
    Dispatches the chat request to the appropriate LLM backend based on the model name or prefix.
    """
    if model.startswith("gpt-"):
        return openai_chat(prompt, api_key=api_key, model=model, system_message=system_message)
    else:
        raise ValueError(f"Unsupported model: {model}")