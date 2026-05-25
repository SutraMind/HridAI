
from openai import OpenAI
import json
import re

# ── LM Studio client ───────────────
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME         = "gemma-4-26b-a4b-it"   # must match exactly what LM Studio shows

client = OpenAI(
    base_url = LM_STUDIO_BASE_URL,
    api_key  = "lm-studio"          # LM Studio ignores this but openai package requires it
)


def call_llm_json(prompt: str) -> dict:
    """
    Call local LM Studio model and parse response as JSON.
    Used by: Input Interpreter, Safety Guard.
    """
    try:
        response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system",  "content": "You are a helpful assistant. Always respond with valid JSON only. No markdown. No explanation."},
                {"role": "user",    "content": prompt}
            ],
            temperature = 0.1,
            max_tokens  = 256,
        )
        # Handle both dict-style and object-style responses
        choice = response.choices[0]
        if hasattr(choice, "message"):
            raw = choice.message.content.strip()
        elif isinstance(choice, dict):
            raw = choice["message"]["content"].strip()
        else:
            raw = str(choice).strip()

        # Strip markdown fences if model adds them
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$",     "", raw)
        raw = re.sub(r"^```\s*",     "", raw)
        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[LLMClient] JSON parse error: {e}\nRaw output: {raw}")
        return {"error": "json_parse", "raw": raw}
    except Exception as e:
        print(f"[LLMClient] Connection error: {e}")
        return {"error": str(e)}


def call_llm_text(prompt: str) -> str:
    """
    Call local LM Studio model and return plain text.
    Used by: Response Generator, Session Summarizer.
    """
    try:
        response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system",  "content": "You are a warm, supportive school robot speaking to a child aged 7-12. Behave like you are his very good friend and make the child feel that the child can share everything to you. Keep responses short and natural."},
                {"role": "user",    "content": prompt}
            ],
            temperature = 0.7,
            max_tokens  = 150,
        )
        # Handle both dict-style and object-style responses
        choice = response.choices[0]
        if hasattr(choice, "message"):
            return choice.message.content.strip()
        elif isinstance(choice, dict):
            return choice["message"]["content"].strip()
        else:
            return str(choice).strip()

    except Exception as e:
        print(f"[LLMClient] Connection error: {e}")
        return "I'm here with you. Can you tell me a bit more?"


# ──test ──────────────────
if __name__ == "__main__":
    print("=== LM Studio Client Test ===\n")

    # Test 1: JSON call
    print("Test 1: JSON structured call")
    json_prompt = """
Return ONLY this exact JSON object, no changes, no markdown:
{"status": "ok", "model": "lmstudio", "connected": true}
"""
    result = call_llm_json(json_prompt)
    print(f"  Result : {result}")
    print(f"  Status : {'✅ OK' if 'error' not in result else '❌ FAILED'}\n")

    # Test 2: Text generation
    print("Test 2: Text generation call")
    text_result = call_llm_text(
        "A child just told you they lost a game and feel upset. "
        "Respond in one warm sentence."
    )
    print(f"  Result : {text_result}")
    print(f"  Status : {'✅ OK' if text_result else '❌ FAILED'}")