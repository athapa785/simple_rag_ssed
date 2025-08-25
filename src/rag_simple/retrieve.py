

from __future__ import annotations
from typing import List, Tuple, Dict, Any

from .config import Config
from .store import get_collection


def retrieve(cfg: Config, question: str) -> List[Dict[str, Any]]:
    col, _ = get_collection(cfg)
    res = col.query(query_texts=[question], n_results=cfg.top_k)

    # chroma returns lists for each query; we only do one query
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for d, m, dist in zip(docs, metas, dists):
        item = dict(m)
        item["text"] = d
        item["score"] = dist  # cosine distance; smaller is more similar
        out.append(item)

    # sort by ascending distance (best first)
    out.sort(key=lambda x: x.get("score", 1.0))
    return out


def make_context(snippets: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    # Concatenate snippets until reaching max_chars, keeping source header lines
    pieces = []
    used = 0
    for s in snippets:
        header = f"[source: {s.get('source')} page:{s.get('page', '')} chunk:{s.get('chunk','')}]\n"
        body = s["text"].strip()
        part = header + body + "\n\n"
        if used + len(part) > max_chars:
            break
        pieces.append(part)
        used += len(part)
    return "".join(pieces)