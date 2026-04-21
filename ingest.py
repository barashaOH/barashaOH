"""Ingest local cybersecurity documents into a Chroma vector store.

Usage:
    python ingest.py
    python ingest.py --reset
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "knowledge"
VECTOR_DB_DIR = BASE_DIR / "vectorstore"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local docs into Chroma")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete current collection before adding fresh documents.",
    )
    return parser


def load_documents(doc_path: Path):
    """Load .txt/.md/.pdf files recursively from knowledge folder."""
    loaders = [
        DirectoryLoader(str(doc_path), glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(doc_path), glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(str(doc_path), glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents(DOCS_DIR)
    if not documents:
        print("No documents found in ./knowledge. Add .txt/.md/.pdf files first.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=embed_model)
    vectorstore = Chroma(
        collection_name="cybersec_knowledge",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    if args.reset:
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name="cybersec_knowledge",
            embedding_function=embeddings,
            persist_directory=str(VECTOR_DB_DIR),
        )

    vectorstore.add_documents(chunks)
    print(
        f"Ingested {len(documents)} documents into {len(chunks)} chunks "
        f"using embedding model '{embed_model}'."
    )


if __name__ == "__main__":
    main()
