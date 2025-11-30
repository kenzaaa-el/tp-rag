import requests
import json

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def hyde_generation(query: str) -> str:
    prompt = f"""
    Generate a hypothetical answer to the question.
    This answer will be used as a synthetic document for dense retrieval.
    Return ONLY the text.

    Question: {query}
    """

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload)
    )

    return response.json()["choices"][0]["message"]["content"]
