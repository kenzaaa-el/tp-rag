# chunking/indexer.py
import os
import json
import pickle
import requests
import numpy as np
from typing import List, Dict

from rank_bm25 import BM25Okapi
from chromadb import PersistentClient

# -------------------------------
# 🔑 OpenRouter API
# -------------------------------
OPENROUTER_API_KEY = "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"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"


class Indexer:
    def __init__(
        self,
        embedding_model="openai/text-embedding-3-small",
        chroma_dir="data/vector_store/",
        bm25_path="data/bm25_index/bm25.pkl",
        metadata_path="data/bm25_index/metadata.pkl"
    ):
        self.embedding_model = embedding_model

        self.chroma_dir = chroma_dir
        self.bm25_path = bm25_path
        self.metadata_path = metadata_path

        os.makedirs(chroma_dir, exist_ok=True)
        os.makedirs(os.path.dirname(bm25_path), exist_ok=True)

        # Chroma persistent client (modern API)
        self.client = PersistentClient(path=chroma_dir)

        self.collection = None
        self.metadata = []      # list of chunk dicts
        self.bm25_index = None


    # ---------------------------------------------------------
    # LOAD ALL CHUNKS
    # ---------------------------------------------------------
    def load_chunks(self, chunks_dir: str) -> List[Dict]:
        chunks = []
        for file in os.listdir(chunks_dir):
            if file.endswith(".json"):
                with open(os.path.join(chunks_dir, file), "r", encoding="utf-8") as f:
                    chunks.extend(json.load(f))

        self.metadata = chunks
        print(f"✔ Loaded {len(chunks)} chunks.")
        return chunks


    # ---------------------------------------------------------
    # EMBEDDINGS via OPENROUTER
    # ---------------------------------------------------------
    def compute_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        texts = [c["text"] for c in chunks]
        all_embeddings = []

        BATCH_SIZE = 80  # safe limit for OpenRouter

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]

            response = requests.post(
                OPENROUTER_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.embedding_model,
                    "input": batch
                }
            )

            data = response.json()

            if "data" not in data:
                print("❌ ERROR in embedding API:", data)
                raise ValueError("OpenRouter embedding failed")

            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)

        print(f"✔ Embeddings generated: {len(all_embeddings)} vectors")
        return all_embeddings


    # ---------------------------------------------------------
    # BUILD CHROMA VECTOR STORE
    # ---------------------------------------------------------
    def build_vector_store(self, chunks: List[Dict], embeddings: List[List[float]]):
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Delete old collection if exists
        if "chunks" in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection("chunks")

        self.collection = self.client.create_collection(name="chunks")

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print("✔ Chroma vector store built & stored")


    # ---------------------------------------------------------
    # BM25 INDEX
    # ---------------------------------------------------------
    def build_bm25_index(self, chunks: List[Dict]):
        tokenized = [c["text"].split() for c in chunks]
        self.bm25_index = BM25Okapi(tokenized)
        print("✔ BM25 index ready")


    # ---------------------------------------------------------
    # SAVE INDEXES (BM25 + metadata)
    # ---------------------------------------------------------
    def save_indexes(self):
        print(f"✔ ChromaDB persisted automatically → {self.chroma_dir}")

        # Save BM25 index
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25_index, f)
        print(f"✔ BM25 index saved → {self.bm25_path}")

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"✔ Metadata saved → {self.metadata_path}")


    # ---------------------------------------------------------
    # LOAD INDEXES (BM25 + metadata)
    # ---------------------------------------------------------
    def load_indexes(self):
        # Load Chroma
        self.client = PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_collection("chunks")

        # Load BM25
        with open(self.bm25_path, "rb") as f:
            self.bm25_index = pickle.load(f)

        # Load metadata
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print("✔ Loaded Chroma + BM25 + metadata successfully")


    # ---------------------------------------------------------
    # SEARCH DENSE (CHROMA)
    # ---------------------------------------------------------
    def search_dense(self, query: str, k: int = 5):
        # Embed query
        response = requests.post(
            OPENROUTER_EMBED_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.embedding_model,
                "input": [query]
            }
        )
        query_embed = response.json()["data"][0]["embedding"]

        results = self.collection.query(
            query_embeddings=[query_embed],
            n_results=k
        )

        return results


    # ---------------------------------------------------------
    # SEARCH SPARSE (BM25)
    # ---------------------------------------------------------
    def search_bm25(self, query: str, k: int = 5):
        scores = self.bm25_index.get_scores(query.split())
        top_k = np.argsort(scores)[::-1][:k]

        # FIXED: now metadata always loaded -> no crash
        return [self.metadata[i] for i in top_k]
