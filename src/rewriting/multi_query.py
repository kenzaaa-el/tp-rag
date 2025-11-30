# src/rewriting/multi_query.py

import requests
import json
import re

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-4o-mini"


# -----------------------------
# JSON EXTRACTION (ROBUST)
# -----------------------------
def extract_json(raw: str):
    """
    Extract a JSON array robustly from messy LLM output.
    Accepts:
        - Perfect JSON
        - JSON with text around
        - JSON with trailing commas
        - Bullet lists fallback
    """

    # Try direct parse
    try:
        return json.loads(raw)
    except:
        pass

    # Extract any JSON-like array inside text
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        cleaned = match.group(0)
        cleaned = re.sub(r",\s*]", "]", cleaned)  # remove trailing comma

        try:
            return json.loads(cleaned)
        except:
            pass

    # Fallback: treat lines as separate rewrites
    lines = [l.strip("-• ") for l in raw.split("\n") if len(l.strip()) > 0]
    return lines[:5]


# -----------------------------
# MULTI QUERY REWRITING
# -----------------------------
def multi_query_rewriting(query: str):
    """
    Produces 5 rewritten variants of the input query.
    """

    prompt = f"""
Rewrite the following query into 5 different search-friendly variants.
Return ONLY a JSON array.

Example:
["rewrite A", "rewrite B", "rewrite C", "rewrite D", "rewrite E"]

Query: {query}
"""

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": prompt}
            ]
        }
    )

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except:
        return [query]  # safe fallback

    rewrites = extract_json(content)

    if not isinstance(rewrites, list) or len(rewrites) == 0:
        return [query]

    return rewrites[:5]
