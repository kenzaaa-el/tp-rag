# src/evaluation/evaluator.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class RAGEvaluator:
    """
    Évalue la qualité d'une réponse générée en comparant :
    - Similarité entre la réponse et les chunks récupérés
    - Longueur et complétude
    - Présence de justification ancrée dans les sources
    """

    def __init__(self):
        pass

    # -------------------------------------------------------------
    # 1. COSINE SIMILARITY ENTRE RÉPONSE ET CHUNKS
    # -------------------------------------------------------------
    def relevance_score(self, answer_embedding, chunks_embeddings):
        """
        Retourne la moyenne de similarité entre la réponse
        et les chunks utilisés.
        """
        sims = cosine_similarity(
            [answer_embedding],
            chunks_embeddings
        )[0]
        return float(np.mean(sims))

    # -------------------------------------------------------------
    # 2. LONGUEUR (SUPPORT)
    # -------------------------------------------------------------
    def support_score(self, answer: str):
        """
        Score basé sur la densité d'informations.
        """
        words = len(answer.split())
        return min(1.0, words / 120)

    # -------------------------------------------------------------
    # 3. SCORE GLOBAL
    # -------------------------------------------------------------
    def evaluate(self, answer, answer_embedding, chunk_texts, chunk_embeddings):
        """
        Score final entre 0 et 1
        """
        rel = self.relevance_score(answer_embedding, chunk_embeddings)
        sup = self.support_score(answer)

        final_score = (0.7 * rel) + (0.3 * sup)

        return {
            "relevance": rel,
            "support": sup,
            "final_score": final_score
        }
