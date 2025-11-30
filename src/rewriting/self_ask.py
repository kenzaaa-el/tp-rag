import requests
import json
import re

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def extract_json(text: str):
    """
    Extracts a JSON array from a messy model output using regex.
    Returns None if no JSON array found.
    """
    match = re.search(r"\[(.|\n)*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None
    return None


def self_ask_decomposition(query: str) -> list:
    prompt = f"""
    Decompose the user's question into 3 to 6 sub-questions needed to answer it.
    RESPONSE RULES:
    - Return ONLY a JSON list (example: ["q1", "q2"])
    - NO explanations, NO prose, NO natural language.
    - JSON ONLY.

    User query: {query}
    """

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload)
    )

    raw_output = response.json()["choices"][0]["message"]["content"]

    # Try a direct JSON parse
    try:
        return json.loads(raw_output)
    except:
        pass

    # Try regex extraction if model wrapped JSON with text
    extracted = extract_json(raw_output)
    if extracted is not None:
        return extracted

    # If everything fails, return fallback minimal decomposition
    return [
        f"What are the causes of the issue: {query}?",
        f"What psychological mechanisms sustain it?",
        f"What patterns or traumas may explain it?"
    ]
