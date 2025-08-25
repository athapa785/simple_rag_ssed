from __future__ import annotations
from typing import List, Tuple


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple length-based chunking with overlap. Works well enough for engineering docs
    without requiring tokenizers.
    """
    if not text:
        return []
    text = text.strip()
    n = len(text)
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, n, step):
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
    return chunks


def attach_metadata(chunks: List[str], base_meta: dict) -> List[Tuple[str, dict]]:
    out = []
    for i, ch in enumerate(chunks):
        meta = dict(base_meta)
        meta["chunk"] = i + 1
        meta["char_len"] = len(ch)
        out.append((ch, meta))
    return out