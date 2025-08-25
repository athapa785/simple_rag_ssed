import os, sys, shutil, time
from dataclasses import replace

# src/ layout bootstrap
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import streamlit as st

from rag_simple.config import Config
from rag_simple.ingest import ingest_dir
from rag_simple.generate import answer
from rag_simple.store import get_collection

DOCS_DIR_DEFAULT = os.path.join(ROOT, "docs")

st.set_page_config(page_title="SSED Document Assistant", layout="wide", page_icon=None)

# Keep chat input fixed at the bottom and avoid overlap with content
st.markdown(
    """
    <style>
    .main .block-container {
        display: grid;
        grid-template-rows: 1fr auto;
        min-height: 100vh;         /* fill viewport height */
    }

    /* Put the input back in normal flow (no fixed/sticky) */
    [data-testid="stChatInput"] {
        position: static;
        margin-top: .75rem;
    }

    /* If your chat area needs its own scrolling instead of the page: */
    .chat-scroll {
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def get_cfg():
    return Config()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def index_count(cfg: Config) -> int:
    try:
        col, _ = get_collection(cfg)
        return col.count()
    except Exception as e:
        return -1

def clear_index(cfg: Config):
    # nukes the Chroma persistent dir
    if os.path.isdir(cfg.db_dir):
        shutil.rmtree(cfg.db_dir)
    # small sleep to avoid file-lock races on some OSes
    time.sleep(0.2)

def save_uploads(files, dest_dir: str):
    ensure_dir(dest_dir)
    saved = []
    for f in files or []:
        path = os.path.join(dest_dir, f.name)
        base, ext = os.path.splitext(path)
        i = 1
        # avoid clobbering existing files
        while os.path.exists(path):
            path = f"{base} ({i}){ext}"
            i += 1
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved.append(path)
    return saved

def sidebar_controls():
    st.sidebar.header("Expert Settings")

    cfg0 = get_cfg()

    docs_dir = st.sidebar.text_input("Docs directory", DOCS_DIR_DEFAULT)
    db_dir = st.sidebar.text_input("Vector DB directory", cfg0.db_dir)
    ensure_dir(docs_dir)
    ensure_dir(db_dir)

    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=cfg0.top_k, step=1)
    chunk_size = st.sidebar.slider("Chunk size (chars)", 500, 3000, cfg0.chunk_size, 100)
    chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 0, 1000, cfg0.chunk_overlap, 50)

    embed_model = st.sidebar.text_input("Embedding model", cfg0.embed_model)
    ollama_host = st.sidebar.text_input("Ollama host", cfg0.ollama_host)
    ollama_model = st.sidebar.text_input("Ollama model", cfg0.ollama_model)


    # create a derived Config with overrides (dataclass replace)
    cfg = replace(
        cfg0,
        db_dir=db_dir,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )
    return cfg, docs_dir

def ui():
    st.title("SSED Document Assistant")

    cfg, docs_dir = sidebar_controls()


    tab_ask, tab_retrain = st.tabs(["Ask", "Re-train (Expert only)"])  # two-tab layout

    # -------------------- Ask tab --------------------
    with tab_ask:
        st.subheader("Ask")

        # Initialize persistent chat history (store both user and assistant messages)
        if "chat" not in st.session_state:
            st.session_state["chat"] = []  # list of {role: "user"|"assistant", content: str, sources?: list}

        # Create two containers so we can render messages ABOVE and keep the input BELOW
        history = st.container()
        input_area = st.container()

        # 1) Render history (above)
        with history:
            for msg in st.session_state["chat"]:
                role = msg.get("role", "assistant")
                with st.chat_message(role):
                    st.markdown(msg.get("content", ""))
                    if role == "assistant" and msg.get("sources"):
                        with st.expander("References"):
                            for s in msg["sources"]:
                                path = s.get("source")
                                page = s.get("page")
                                score = s.get("score")
                                base = os.path.basename(path) if path else ""
                                label = (f"• {base} — page {page}, dist {score:.4f}" if score is not None else f"• {base} — page {page}")
                                st.markdown(label)

        # 2) Input stays at the bottom
        with input_area:
            user_q = st.chat_input("Type your question…")

        # 3) If user asked something, render Q → Fetching → (append answer) into HISTORY, then rerun
        if user_q:
            # Persist user question
            st.session_state["chat"].append({"role": "user", "content": user_q})

            with history:
                # Show user question alone
                with st.chat_message("user"):
                    st.markdown(user_q)

                # Show only a spinner while we compute (no temporary text block)
                with st.chat_message("assistant"):
                    with st.spinner("Working…"):
                        resp = answer(cfg, user_q.strip())

            # Persist assistant message so it renders once in history on rerun
            st.session_state["chat"].append({
                "role": "assistant",
                "content": resp.get("answer", ""),
                "sources": resp.get("sources", []),
            })

            # Rerun to display the newly appended assistant message without duplicates
            st.rerun()
    # -------------------- Retrain tab --------------------
    with tab_retrain:
        st.subheader("Index")
        c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="center")
        with c1:
            if st.button("Refresh index size", use_container_width=True, key="btn_refresh_index"):
                st.rerun()
        with c2:
            if st.button("Clear index (delete DB)", type="secondary", use_container_width=True, key="btn_clear_index"):
                clear_index(cfg)
                st.success("Vector DB cleared.")
                # Log to chat
                if "chat" not in st.session_state:
                    st.session_state["chat"] = []
                st.session_state["chat"].append({
                    "role": "assistant",
                    "content": "**Index action**\n\nVector DB cleared.",
                    "sources": [],
                })
                st.rerun()
        with c3:
            count = index_count(cfg)
            if count >= 0:
                st.info(
                    f"Current collection: **{cfg.collection}**  |  DB dir: `{cfg.db_dir}`  |  **{count}** chunks indexed"
                )
            else:
                st.warning("Index not initialized yet.")

        st.subheader("Add Documents")
        uploaded = st.file_uploader(
            "Drop PDFs / images (.png/.jpg/.jpeg/.tif/.tiff) / text (.txt/.md)",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "txt", "md"],
            accept_multiple_files=True,
            key="uploader_docs",
        )
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            if st.button("Save uploads to docs/ and Ingest", type="primary", key="btn_upload_ingest"):
                paths = save_uploads(uploaded, docs_dir)
                if paths:
                    with st.spinner("Ingesting..."):
                        ingest_dir(cfg, docs_dir)
                    st.success(f"Ingested {len(paths)} file(s).")
                    # Log to chat
                    new_count = index_count(cfg)
                    if "chat" not in st.session_state:
                        st.session_state["chat"] = []
                    st.session_state["chat"].append({
                        "role": "assistant",
                        "content": f"**Index action**\n\nSaved and ingested {len(paths)} file(s). Collection size: **{new_count}**.",
                        "sources": [],
                    })
                    st.rerun()
                else:
                    st.info("Nothing uploaded.")
        with col_u2:
            if st.button(
                "Rebuild from docs/ (quick add)",
                help=(
                    "Ingests all files under the docs directory. If you already indexed them before, "
                    "consider clearing the index to avoid duplicate-id errors."
                ),
                key="btn_rebuild_docs",
            ):
                with st.spinner("Ingesting docs/ ..."):
                    ingest_dir(cfg, docs_dir)
                st.success("Ingestion finished.")
                # Log to chat
                new_count = index_count(cfg)
                if "chat" not in st.session_state:
                    st.session_state["chat"] = []
                st.session_state["chat"].append({
                    "role": "assistant",
                    "content": f"**Index action**\n\nRe-ingested docs/. Collection size: **{new_count}**.",
                    "sources": [],
                })
                st.rerun()

        st.caption(f"Docs dir: `{docs_dir}`  – You can also populate it manually in Finder/Explorer.")

if __name__ == "__main__":
    ui()