# src/retrieval/rrf.py
from typing import List, Dict, Any


def rrf_fusion(
    result_sets: List[List[Dict[str, Any]]],
    k: int = 5,
    alpha: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) pour fusionner plusieurs listes de résultats.

    :param result_sets: liste de listes de documents.
                        Chaque document DOIT contenir une clé "id".
    :param k: nombre final de documents à retourner
    :param alpha: paramètre de lissage (par défaut 60)
    :return: liste fusionnée de documents (top-k)
    """
    scores = {}

    for results in result_sets:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            increment = 1.0 / (alpha + rank + 1)

            if doc_id not in scores:
                scores[doc_id] = {
                    "doc": doc,
                    "score": 0.0,
                }

            scores[doc_id]["score"] += increment

    # tri décroissant de score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    fused_docs = [entry["doc"] for entry in ranked[:k]]
    return fused_docs
