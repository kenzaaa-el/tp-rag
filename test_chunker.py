# tests/test_chunker_data.py
import os
import glob
from src.chunking.chunker import DocumentChunker

def test_chunker_on_data():
    data_dir = "data/"
    output_dir = "data/chunks/"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDFs
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    assert len(pdf_files) > 0, "No PDF files found in /data"

    print(f"Found {len(pdf_files)} PDFs in /data")

    # Initialize chunker
    chunker = DocumentChunker(chunk_size=600, overlap=100)

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n🔍 Processing: {pdf_name}")

        chunks = chunker.process_document(
            file_path=pdf_path,
            output_dir=output_dir
        )

        print(f"✔ {pdf_name} → {len(chunks)} chunks generated.")

    print("\n🎉 All PDFs processed successfully!")


if __name__ == "__main__":
    test_chunker_on_data()
