# cli.py
import argparse
import yaml
from src.pipeline_rag import AdvancedRAG

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Advanced RAG from CLI")
    parser.add_argument("--query", type=str, required=True, help="User question")

    args = parser.parse_args()
    query = args.query

    config = load_config()
    rag = AdvancedRAG(config)

    result = rag.run(query)

    print("\n" + "="*80)
    print("ANSWER:\n")
    print(result["answer"])
    print("="*80)

    # -----------------------------
    # PRINT EVALUATION SCORES
    # -----------------------------
    eval = result["evaluation"]

    print("\nEVALUATION SCORES")
    print("-" * 80)
    print(f"Relevance Score : {eval['relevance']:.4f}")
    print(f"Support Score   : {eval['support']:.4f}")
    print(f"Final Score     : {eval['final_score']:.4f}")
    print("-" * 80)

    print("\nDone.")

if __name__ == "__main__":
    main()
