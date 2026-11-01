"""Beginner entry point; reusable PDF extraction lives in finrag.ingest."""
from finrag.config import Settings
from finrag.ingest import read_corpus

if __name__ == "__main__":
    pages, _ = read_corpus(Settings.load().root)
    print(f"The document contains {len(pages)} pages.\n")
    print("Extracted text from page 1:\n")
    print(pages[0].page_content)
