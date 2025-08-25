

#!/usr/bin/env python3
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from rag_simple.config import Config
from rag_simple.generate import answer

app = FastAPI(title="Simple Dense RAG API")


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/ask")
def ask(q: str = Query(..., description="User question")):
    cfg = Config()
    resp = answer(cfg, q)
    return JSONResponse(resp)