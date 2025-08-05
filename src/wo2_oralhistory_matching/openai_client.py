import openai
import time
import re

def _chat(prompt: str, api_key: str, model: str = "gpt-4.1", system_message: str = "You help to segment interviews.") -> str:
    """
    Creates a chat-query for the OpenAI API.
    """
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=16384
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"OpenAI API fout: {e}")
        raise

def _safe_chat_call(prompt: str, api_key: str, model: str = "gpt-4.1", system_message: str = "You help to segment interviews.", max_retries: int = 5) -> str:
    """
    Calls _chat with intelligent retry and delay if a rate limit is hit.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            return _chat(prompt, api_key=api_key, model=model, system_message=system_message)
        except openai.RateLimitError as e:
            error_msg = str(e)
            print("[RateLimitError] Rate limit reached.")
            wait_match = re.search(r"try again in ([\d\.]+)s", error_msg)
            if wait_match:
                wait_time = float(wait_match.group(1))
            else:
                wait_time = 10 
                print("Could not find waiting time, using 10 seconds.")
            print(f"Waiting {wait_time} seconds for next attempt...")
            time.sleep(wait_time)
            attempt += 1
        except openai.OpenAIError as e:
            print(f"[OpenAIError] {e}")
            raise
    raise Exception("Too many attempts, even with time handling.")