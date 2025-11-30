# src/retrieval/bm25_retriever.py
from typing import List, Dict, Any


class BM25Retriever:
    """
    Retriever lexical BM25.
    Utilise Indexer.search_bm25(query, k) qui renvoie directement les chunks.
    """

    def __init__(self, indexer):
        """
        :param indexer: instance de Indexer
        """
        self.indexer = indexer

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Lance une recherche BM25 sur les chunks tokenisés.

        :param query: chaîne de caractères (requête)
        :param k: nombre de documents à retourner
        :return: liste de dicts {"id", "text", "metadata"}
        """
        bm25_results = self.indexer.search_bm25(query, k=k)

        # On s'assure d'un format homogène avec DenseRetriever
        documents: List[Dict[str, Any]] = []
        for chunk in bm25_results:
            documents.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "distance": None,  # pas de notion de distance, mais on garde la clé
            })

        return documents
