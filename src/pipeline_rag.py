# src/pipeline_rag.py

from src.rewriting.apply_method import apply_rewriting_method
from src.chunking.indexer import Indexer
from src.retrieval.retrieval_controller import RetrievalController
from src.generation.generator import AnswerGenerator
from src.evaluation.evaluator import RAGEvaluator
import requests


OPENROUTER_API_KEY = "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"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL_EMBED = "openai/text-embedding-3-small"


class AdvancedRAG:
    def __init__(self, config):
        self.config = config
        self.indexer = Indexer()
        self.indexer.load_indexes()

        self.retriever = RetrievalController(self.indexer)
        self.generator = AnswerGenerator()
        self.evaluator = RAGEvaluator()     # NEW

    # ------------------------
    # Helper: embed answer
    # ------------------------
    def embed_text(self, text: str):
        resp = requests.post(
            OPENROUTER_EMBED_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_EMBED,
                "input": [text]
            }
        )
        return resp.json()["data"][0]["embedding"]

    # ------------------------
    # FULL PIPELINE
    # ------------------------
    def run(self, query: str, k: int = 5):
        # Step 1 — rewriting
        rewriting_output = apply_rewriting_method(query)

        # Step 2 — retrieval
        retrieved_docs = self.retriever.run(rewriting_output, k=k)

        # Step 3 — answer generation
        final_answer = self.generator.generate(query, retrieved_docs)

        # Step 4 — evaluation
        chunk_texts = [doc["text"] for doc in retrieved_docs]

        # Embed answer
        answer_emb = self.embed_text(final_answer)

        # Embed chunks (we already have them from Indexer)
        chunk_embs = [self.embed_text(doc["text"]) for doc in retrieved_docs]

        evaluation_scores = self.evaluator.evaluate(
            answer=final_answer,
            answer_embedding=answer_emb,
            chunk_texts=chunk_texts,
            chunk_embeddings=chunk_embs
        )

        # Output
        return {
            "answer": final_answer,
            "rewriting_output": rewriting_output,
            "retrieved_docs": retrieved_docs,
            "evaluation": evaluation_scores         # NEW
        }
