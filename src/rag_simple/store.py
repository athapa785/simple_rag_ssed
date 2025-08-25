from __future__ import annotations
import os
import chromadb
from chromadb.utils import embedding_functions

from .config import Config


def get_collection(cfg: Config):
    os.makedirs(cfg.db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=cfg.db_dir)  # 0.5+

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.embed_model,
        normalize_embeddings=True,
    )

    col = client.get_or_create_collection(
        name=cfg.collection,
        embedding_function=ef,
    )
    return col, client