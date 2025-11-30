import os
import json
import re
from typing import List, Dict
import pdfplumber


class DocumentChunker:
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # -----------------------------------------------------------
    # 1. LOAD DOCUMENT
    # -----------------------------------------------------------
    def load_document(self, file_path: str) -> str:
        """Extract raw text from a PDF or text file."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".txt":
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += "\n" + page_text
        return text

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # -----------------------------------------------------------
    # 2. CLEAN TEXT
    # -----------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Normalize text: remove spaces, extra newlines, weird chars."""
        text = re.sub(r'\s+', ' ', text)  # collapse whitespace
        text = text.strip()
        return text

    # -----------------------------------------------------------
    # 3. SPLIT INTO CHUNKS
    # -----------------------------------------------------------
    def split_into_chunks(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        chunk_id = 1

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "metadata": {
                    "word_start": start,
                    "word_end": end
                }
            })

            chunk_id += 1
            start += (self.chunk_size - self.overlap)

        return chunks

    # -----------------------------------------------------------
    # 4. SAVE CHUNKS
    # -----------------------------------------------------------
    def save_chunks(self, chunks: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------
    # 5. FULL PIPELINE
    # -----------------------------------------------------------
    def process_document(self, file_path: str, output_dir: str):
        """Full pipeline: load → clean → chunk → save."""
        raw_text = self.load_document(file_path)
        cleaned_text = self.clean_text(raw_text)
        chunks = self.split_into_chunks(cleaned_text)

        out_path = os.path.join(output_dir, "chunks.json")
        self.save_chunks(chunks, out_path)

        print(f"✔ Chunking complete: {len(chunks)} chunks saved to {out_path}")
        return chunks
