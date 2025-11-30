# src/retrieval/dense_retriever.py
from typing import List, Dict, Any


class DenseRetriever:
    """
    Dense retriever basé sur Chroma + embeddings OpenRouter.
    Utilise Indexer.search_dense(query, k).
    """

    def __init__(self, indexer):
        """
        :param indexer: instance de Indexer (src.chunking.indexer.Indexer)
        """
        self.indexer = indexer

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Lance une dense search sur Chroma pour une requête texte.

        :param query: chaîne de caractères (requête, sous-question, doc HyDE, etc.)
        :param k: nombre de documents à retourner 5 par défaut
        :return: liste de dicts {"id", "text", "metadata", "distance"}
        """
        results = self.indexer.search_dense(query, k=k)

        documents: List[Dict[str, Any]] = []

        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])
        distances = distances[0] if distances else [None] * len(docs)

        for i in range(len(docs)):
            documents.append({
                "id": ids[i],
                "text": docs[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            })

        return documents
