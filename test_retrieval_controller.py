from src.chunking.indexer import Indexer
from src.retrieval import RetrievalController


def test_retrieval_controller():
    indexer = Indexer()
    indexer.load_indexes()

    controller = RetrievalController(indexer)

    rewriting_output = {
        "mode": "none",
        "query": "Why do I fear abandonment?"
    }

    results = controller.run(rewriting_output, k=5)

    print("\n[RetrievalController] Results:")
    for i, doc in enumerate(results, start=1):
        print(f"\n--- #{i} ---")
        print("ID:", doc["id"])
        print("Text:", doc["text"][:250], "...")
        print("Metadata:", doc["metadata"])

if __name__ == "__main__":
    test_retrieval_controller()