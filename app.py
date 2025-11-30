# app.py
import streamlit as st
from src.pipeline_rag import AdvancedRAG
import yaml

st.set_page_config(page_title="Advanced RAG System", layout="wide")

# -----------------------------
# Load YAML config
# -----------------------------
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
rag = AdvancedRAG(config)

# -----------------------------
# UI TITLE
# -----------------------------
st.title("🧠 Advanced Retrieval Augmented Generation (RAG)")
st.markdown("### Powered by OpenRouter + Advanced Multi-Retriever")


# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("Enter your question:", placeholder="e.g., What is cognitive dissonance?")

if st.button("Run RAG"):

    if not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Running RAG pipeline..."):
            result = rag.run(query)

        # -----------------------------
        # ANSWER SECTION
        # -----------------------------
        st.subheader("🟦 Final Answer")
        st.write(result["answer"])

        # -----------------------------
        # REWRITING DETAILS
        # -----------------------------
        st.subheader("🟧 Rewriting Output")
        st.json(result["rewriting_output"])

        # -----------------------------
        # EVALUATION
        # -----------------------------
        st.subheader("🟩 Evaluation Scores")

        eval_scores = result["evaluation"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Relevance", f"{eval_scores['relevance']:.4f}")
        col2.metric("Support", f"{eval_scores['support']:.4f}")
        col3.metric("Final Score", f"{eval_scores['final_score']:.4f}")

        # -----------------------------
        # RETRIEVED DOCUMENTS
        # -----------------------------
        st.subheader("📄 Retrieved Chunks")

        for i, doc in enumerate(result["retrieved_docs"], start=1):
            with st.expander(f"Chunk {i} – ID={doc['id']}"):
                st.write(doc["text"])
                st.json(doc["metadata"])
