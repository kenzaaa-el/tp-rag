# tests/test_indexer.py
from src.chunking import Indexer

def test_indexer():
    indexer = Indexer()

    chunks = indexer.load_chunks("data/chunks/")
    embeddings = indexer.compute_embeddings(chunks)

    indexer.build_vector_store(chunks, embeddings)
    indexer.build_bm25_index(chunks)

    indexer.save_indexes()

    print("🎉 Indexer test passed!")


if __name__ == "__main__":
    test_indexer()
