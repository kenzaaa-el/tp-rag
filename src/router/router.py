import requests
import json

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def query_router(user_query: str) -> dict:
    """
    Route la question utilisateur vers :
      - multi_query
      - hyde
      - self_ask
      - none
    Retour : { "type": "...", "reasoning": "..." }
    """

    system_prompt = """
    You are a Query Router for an Advanced RAG system.
    Your job is to classify the user query into EXACTLY one of these categories:

      1. multi_query  → simple question, needs reformulations
      2. hyde         → vague or short question, needs hypothetical document
      3. self_ask     → complex question, multi-step, needs decomposition
      4. none         → no rewriting needed

    Return ONLY JSON with fields:
        type: string
        reasoning: concise explanation

    Example output:
    {"type": "hyde", "reasoning": "..."}
    """

    payload = {
        "model": "openai/gpt-4o-mini",    
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
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

    data = response.json()
    message = data["choices"][0]["message"]["content"]

    # Normalisation et parsing JSON
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return {"type": "none", "reasoning": "Failed to parse model output."}


# 🔍 Petite démo
if __name__ == "__main__":
    examples = [
        "Pourquoi j’ai peur qu’on me quitte ?",
        "Pourquoi je répète des relations toxiques ?",
        "Qu’est-ce que l’attachement anxieux ?",
        "Pourquoi je me sens vide parfois ?",
    ]

    for q in examples:
        print("USER:", q)
        print("ROUTER:", query_router(q))
        print("-" * 70)
