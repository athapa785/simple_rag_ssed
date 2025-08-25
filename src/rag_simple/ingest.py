

from __future__ import annotations
import os
import glob
import hashlib
from typing import List

from tqdm import tqdm

from .config import Config
from .logging_setup import logger
from .text_extractor import iter_docs
from .chunker import chunk_text, attach_metadata
from .store import get_collection


SUPPORTED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt", ".md")


def _doc_paths(root: str) -> List[str]:
    root = os.path.abspath(root)
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(set(paths))


def _id_for(path: str, unit_id: str, chunk_idx: int) -> str:
    raw = f"{unit_id}:{chunk_idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def ingest_dir(cfg: Config, docs_dir: str):
    col, client = get_collection(cfg)

    paths = _doc_paths(docs_dir)
    if not paths:
        logger.warning(f"No supported documents found in {docs_dir}")
        return

    logger.info(f"Found {len(paths)} files. Ingesting → {cfg.db_dir} / {cfg.collection}")

    batch_ids, batch_docs, batch_metas = [], [], []
    BATCH = 128  # small batches to keep memory low

    for pth in tqdm(paths, desc="files"):
        for unit_id, text, meta in iter_docs(pth):
            chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
            for i, (chunk, m) in enumerate(attach_metadata(chunks, meta)):
                uid = _id_for(pth, unit_id, i)
                batch_ids.append(uid)
                batch_docs.append(chunk)
                batch_metas.append(m)
                if len(batch_ids) >= BATCH:
                    col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                    batch_ids, batch_docs, batch_metas = [], [], []
    if batch_ids:
        col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    # Persist

    count = col.count()
    logger.info(f"Ingestion complete. Collection size: {count}")