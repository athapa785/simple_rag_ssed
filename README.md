# Simple, Robust RAG (dense-only)

This is a **single-method RAG**: *dense vector retrieval* (Chroma + Sentence-Transformers) → prompt stuffing → local LLM (Ollama). It's intentionally simple but still robust:

- **RAM friendly**: on-disk Chroma, incremental ingestion, small embedding model by default.
- **Diagrams/drawings support**: optional OCR via Tesseract (if installed) using PyMuPDF page rasterization only when a page has no text.
- **No rerankers, no two-stage retrievers**: fewer moving parts, fewer failure modes.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that:

1. Indexes PDF documents, images, and text files using dense vector embeddings
2. Processes user questions by retrieving relevant document chunks
3. Generates answers using local LLMs through Ollama

### Architecture

- **Document Processing**: PDF parsing with PyMuPDF, OCR with Tesseract when needed
- **Embedding**: Sentence-Transformers (default: BAAI/bge-small-en-v1.5)
- **Vector Database**: Chroma (persistent on-disk storage)
- **Generation**: Ollama with local models (default: llama3.1:8b)
- **API**: FastAPI server for integration with other systems

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple_rag_ssed.git
cd simple_rag_ssed

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# (Optional) Install Tesseract for OCR support
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

# (Optional) Install Ollama for local LLM generation
# https://ollama.com/download
ollama pull llama3.1:8b
```

## Usage

### 1. Document Ingestion

Place your documents in the `./docs` directory. Supported formats:
- PDF files (.pdf)
- Images (.png, .jpg, .jpeg, .tif, .tiff)
- Text files (.txt, .md)

Then index them:

```bash
python scripts/build_index.py --docs ./docs
```

### 2. Ask Questions (CLI)

```bash
python scripts/ask.py "Your question about the documents"
```

### 3. Run as API Server

```bash
# Use the venv's interpreter to avoid import issues
python -m uvicorn scripts.serve:app --host 0.0.0.0 --port 8080
# or explicitly use the venv's uvicorn binary
./.venv/bin/uvicorn scripts.serve:app --host 0.0.0.0 --port 8080
```

> If you accidentally run the global/conda uvicorn (e.g., /opt/anaconda3/bin/uvicorn), it won't see the project's src/ package. Always use the venv's uvicorn or `python -m uvicorn`.

Then query using:

```bash
curl 'http://localhost:8080/ask?q=Your%20question%20about%20the%20documents'
```

Or access the API docs at: http://localhost:8080/docs

### 4. Run Streamlit UI

The project includes a user-friendly Streamlit interface for document management and querying:

```bash
python -m streamlit run app/streamlit_app.py
```

This provides:
- A document upload interface with drag-and-drop functionality
- Interactive question answering with source citations
- Advanced configuration options for embedding models and chunking parameters

## Configuration

The system can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| RAG_EMBED_MODEL | Sentence-Transformer model for embeddings | BAAI/bge-small-en-v1.5 |
| RAG_COLLECTION | Chroma collection name | company_docs |
| RAG_DB_DIR | Directory for vector database | ./vectorstore |
| RAG_CHUNK_SIZE | Document chunk size in characters | 1200 |
| RAG_CHUNK_OVERLAP | Overlap between chunks | 200 |
| RAG_TOP_K | Number of chunks to retrieve | 8 |
| OLLAMA_HOST | Ollama API endpoint | http://localhost:11434 |
| OLLAMA_MODEL | Model to use for generation | llama3.1:8b |

Create a `.env` file in the project root to set these variables:

```
RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5
RAG_COLLECTION=my_docs
RAG_TOP_K=10
OLLAMA_MODEL=mistral:7b
```

## Project Structure

```
simple_rag_ssed/
├── app/                 # Web application
│   └── streamlit_app.py # Streamlit UI
├── docs/                # Place your documents here
├── scripts/             # Command-line scripts
│   ├── ask.py           # CLI question answering
│   ├── build_index.py   # Document indexing
│   └── serve.py         # FastAPI server
├── src/
│   └── rag_simple/      # Core library
│       ├── chunker.py      # Document chunking
│       ├── config.py       # Configuration
│       ├── generate.py     # LLM integration
│       ├── ingest.py       # Document processing
│       ├── retrieve.py     # Vector retrieval
│       ├── store.py        # Chroma integration
│       └── text_extractor.py # PDF/text extraction
├── vectorstore/         # Vector database storage (created on first run)
├── .env                 # Environment variables (create this)
└── requirements.txt     # Python dependencies
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.