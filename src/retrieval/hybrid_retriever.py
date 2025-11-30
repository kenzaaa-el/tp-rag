# src/retrieval/hybrid_retriever.py
from typing import List, Dict, Any
from .rrf import rrf_fusion
from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Combinaison Dense + BM25 avec fusion RRF.
    """

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Lance une recherche hybride sur une seule requête.

        :param query: texte de la requête
        :param k: nombre de documents à retourner
        :return: liste de documents fusionnés Dense+BM25
        """
        dense_results = self.dense.query(query, k=k)
        bm25_results = self.bm25.query(query, k=k)

        fused = rrf_fusion([dense_results, bm25_results], k=k)
        return fused
