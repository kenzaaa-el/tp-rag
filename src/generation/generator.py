# src/generation/generator.py

import requests
from typing import List, Dict, Any

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class AnswerGenerator:
    """
    Turns: query + retrieved chunks → grounded final answer
    using OpenRouter LLM (4o-mini)
    """

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model

    def build_context_block(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a clean context block.
        """
        block = ""
        for i, doc in enumerate(chunks, start=1):
            block += (
                f"[CHUNK {i} — ID={doc['id']}]\n"
                f"{doc['text']}\n\n"
            )
        return block

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Full answer synthesis:
        - format context
        - build the prompt
        - call the LLM
        - return the generated answer
        """
        context_block = self.build_context_block(retrieved_docs)

        prompt = f"""
You are psychologist. A user has asked you a question, and you have access to several context chunks that may help you answer it. 
Use ONLY the information from the CONTEXT to answer the USER QUESTION as accurately as possible. If the CONTEXT does not contain enough information, answer with your own knowledge, but indicate that the information was not found in the CONTEXT."

USER QUESTION:
{query}

CONTEXT:
{context_block}

Now provide your final answer using ONLY the above context.
"""

        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a precise RAG answer generator."},
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except:
            return "❌ LLM generation error: " + str(data)
