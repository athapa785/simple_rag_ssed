"""rag_simple package

Lightweight CLI entry points are provided here so we can expose console scripts
without adding new modules. These are intended for *editable installs* and local
use; for wheels, move the Streamlit app under `src/` later.
"""
from __future__ import annotations

from pathlib import Path
import sys
import subprocess
import argparse

from .config import Config
from .ingest import ingest_dir
from .generate import answer

__all__ = [
    "build_index_cli",
    "ask_cli",
    "serve_cli",
    "ui_cli",
]


def _repo_root_from_pkg() -> Path | None:
    """Best-effort: find the project root (where `app/streamlit_app.py` lives)
    when installed in editable mode with a `src/` layout. Returns None if not found.
    """
    here = Path(__file__).resolve()
    # common src-layout: .../repo/src/rag_simple/__init__.py -> repo
    for up in [here.parent, here.parent.parent, here.parent.parent.parent]:
        # up: rag_simple/, src/, repo/
        candidate = up.parent if up.name == "src" else up
        if (candidate / "app" / "streamlit_app.py").exists():
            return candidate
    # walk up a few more just in case
    p = here
    for _ in range(5):
        if (p / "app" / "streamlit_app.py").exists():
            return p
        p = p.parent
    return None


def build_index_cli() -> None:
    p = argparse.ArgumentParser(description="Ingest documents into Chroma")
    p.add_argument("--docs", default="./docs")
    args = p.parse_args()
    ingest_dir(Config(), args.docs)


def ask_cli() -> None:
    p = argparse.ArgumentParser(description="Ask a question against the index")
    p.add_argument("question")
    args = p.parse_args()
    resp = answer(Config(), args.question)
    print("\n=== ANSWER ===\n")
    print(resp.get("answer", ""))
    print("\n=== SOURCES ===\n")
    for s in resp.get("sources", []):
        src = s.get("source")
        page = s.get("page")
        chunk = s.get("chunk")
        score = s.get("score")
        if score is not None:
            print(f"- {src} (page {page}, chunk {chunk}, dist {score:.4f})")
        else:
            print(f"- {src} (page {page}, chunk {chunk})")


def serve_cli() -> None:
    # Inline FastAPI app to avoid creating new files
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import JSONResponse
        import uvicorn
    except Exception:
        print("FastAPI/uvicorn not installed. Install requirements or run `pip install -r requirements.txt`.",
              file=sys.stderr)
        raise

    app = FastAPI(title="Simple Dense RAG API")

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.get("/ask")
    def ask(q: str = Query(..., description="User question")):
        return JSONResponse(answer(Config(), q))

    uvicorn.run(app, host="0.0.0.0", port=8080)


def ui_cli() -> None:
    """Run the Streamlit UI from the repo (editable install).

    Looks for `app/streamlit_app.py` relative to the installed package. This works
    in editable installs (recommended for development). If you package a wheel,
    move the app under `src/rag_simple/ui/` and update the launcher accordingly.
    """
    root = _repo_root_from_pkg()
    if root is None:
        print(
            "Could not locate app/streamlit_app.py. If you're running from a wheel, "
            "move the app under src/rag_simple/ui and adjust the launcher.",
            file=sys.stderr,
        )
        sys.exit(2)
    app_path = root / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    sys.exit(subprocess.call(cmd))