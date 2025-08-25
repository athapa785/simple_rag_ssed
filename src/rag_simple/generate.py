

from __future__ import annotations
from typing import Dict, Any
import os

from .config import Config
from .retrieve import retrieve, make_context
from .logging_setup import logger

try:
    import ollama
except Exception as e:
    ollama = None


SYS_PROMPT = (
    "You are a careful technical assistant answering ONLY from the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources inline using markers already included in the context."
)


def answer(cfg: Config, question: str) -> Dict[str, Any]:
    snippets = retrieve(cfg, question)
    context = make_context(snippets)

    if not context.strip():
        return {
            "answer": "I don't have enough information in the indexed corpus to answer that.",
            "sources": [],
        }

    prompt = (
        f"System: {SYS_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {question}\n\n"
        f"Answer concisely, and include citations by quoting the headers where relevant."
    )

    if ollama is None:
        logger.warning("ollama is not installed; returning context-only stub answer")
        return {"answer": context[:1200] + "\n\n[Install ollama to generate answers]", "sources": snippets}

    client = ollama.Client(host=cfg.ollama_host)
    resp = client.generate(model=cfg.ollama_model, prompt=prompt, options={"num_ctx": 8192})
    txt = resp.get("response", "")
    return {"answer": txt, "sources": snippets}