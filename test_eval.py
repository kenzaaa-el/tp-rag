# tests/test_evaluator.py

import requests
from src.evaluation.evaluator import RAGEvaluator

OPENROUTER_API_KEY = "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"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "openai/text-embedding-3-small"


def embed_text(text: str):
    """Simple helper to embed one text string."""
    resp = requests.post(
        OPENROUTER_EMBED_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBED_MODEL,
            "input": [text]
        }
    )
    data = resp.json()
    return data["data"][0]["embedding"]


def test_evaluator():
    evaluator = RAGEvaluator()

    # ---------------------------
    # 1. Simulated RAG output
    # ---------------------------
    answer = (
        "Abandonment fears often come from childhood instability, "
        "attachment issues, or inconsistent caregiving."
    )

    retrieved_chunks = [
        "Fear of abandonment is often linked to insecure attachment styles.",
        "Childhood instability and inconsistent nurturing contribute to abandonment anxiety."
    ]

    # ---------------------------
    # 2. Embed answer + chunks
    # ---------------------------
    print("Embedding answer...")
    answer_emb = embed_text(answer)

    print("Embedding chunks...")
    chunk_embs = [embed_text(c) for c in retrieved_chunks]

    # ---------------------------
    # 3. Evaluate
    # ---------------------------
    scores = evaluator.evaluate(
        answer=answer,
        answer_embedding=answer_emb,
        chunk_texts=retrieved_chunks,
        chunk_embeddings=chunk_embs
    )

    print("\n=== Evaluation Result ===")
    print("Relevance Score:", scores["relevance"])
    print("Support Score:", scores["support"])
    print("Final Score:", scores["final_score"])

    print("\n🎉 Evaluator test passed!")


if __name__ == "__main__":
    test_evaluator()
