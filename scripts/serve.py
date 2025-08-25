

#!/usr/bin/env python3
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