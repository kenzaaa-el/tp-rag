# src/retrieval/retrieval_controller.py
from typing import List, Dict, Any

from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .rrf import rrf_fusion


class RetrievalController:
    """
    Orchestrateur de retrieval :
    - choisit la bonne stratégie selon le mode de rewriting
    - utilise l'Indexer pour interroger les indexes
    """

    def __init__(self, indexer):
        """
        :param indexer: instance de Indexer (déjà chargé avec load_indexes())
        """
        self.indexer = indexer

        # Retrievers de base
        self.dense = DenseRetriever(indexer)
        self.bm25 = BM25Retriever(indexer)
        self.hybrid = HybridRetriever(self.dense, self.bm25)

    def run(self, rewriting_output: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """
        Applique la bonne méthode de retrieval en fonction du rewriting.

        :param rewriting_output: dict provenant de apply_rewriting_method(query)
               structure possible :
                - {"mode": "multi_query", "queries": [...]}
                - {"mode": "self_ask", "queries": [...]}
                - {"mode": "hyde", "document": "..."}
                - {"mode": "none", "query": "..."}
        :param k: nombre de documents finaux à retourner
        :return: liste de documents (chunks) prêts pour la génération
        """
        mode = rewriting_output["mode"]

        # 1) MULTI-QUERY → Hybrid + RRF sur chaque requête
        if mode == "multi_query":
            queries = rewriting_output["queries"]
            all_results = []

            for q in queries:
                res = self.hybrid.query(q, k=k)
                all_results.append(res)

            fused = rrf_fusion(all_results, k=k)
            return fused

        # 2) SELF-ASK → Dense-only sur chaque sous-question + RRF
        elif mode == "self_ask":
            subquestions = rewriting_output["queries"]
            all_results = []

            for sq in subquestions:
                res = self.dense.query(sq, k=k)
                all_results.append(res)

            fused = rrf_fusion(all_results, k=k)
            return fused

        # 3) HYDE → Dense-only sur le document synthétique
        elif mode == "hyde":
            synthetic_doc = rewriting_output["document"]
            results = self.dense.query(synthetic_doc, k=k)
            return results

        # 4) AUCUN REWRITING → Hybrid "fusion légère"
        else:
            query = rewriting_output["query"]

            # petite hybrid fusion : priorité au dense, puis BM25 sans RRF
            dense_results = self.dense.query(query, k=min(k, 3))
            bm25_results = self.bm25.query(query, k=min(k, 3))

            seen_ids = set(doc["id"] for doc in dense_results)
            merged = dense_results.copy()

            for doc in bm25_results:
                if doc["id"] not in seen_ids:
                    merged.append(doc)
                    seen_ids.add(doc["id"])
                if len(merged) >= k:
                    break

            return merged[:k]
