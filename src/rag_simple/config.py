import os
from dataclasses import dataclass

@dataclass
class Config:
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    collection: str = os.getenv("RAG_COLLECTION", "company_docs")
    db_dir: str = os.getenv("RAG_DB_DIR", "./vectorstore")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("RAG_TOP_K", "8"))

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Add more knobs if needed later