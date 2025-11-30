from src.pipeline_rag import AdvancedRAG

def test_rag():
    rag = AdvancedRAG(config={})

    query = "What is anxiety ?"
    result = rag.run(query, k=5)

    print("\n===== FINAL ANSWER =====\n")
    print(result["answer"])


if __name__ == "__main__":
    test_rag()
