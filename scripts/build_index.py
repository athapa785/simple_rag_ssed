

#!/usr/bin/env python3
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import argparse
from rag_simple.config import Config
from rag_simple.ingest import ingest_dir


def main():
    p = argparse.ArgumentParser(description="Ingest documents into Chroma")
    p.add_argument("--docs", default="./docs", help="Directory of documents to ingest")
    args = p.parse_args()

    cfg = Config()
    ingest_dir(cfg, args.docs)


if __name__ == "__main__":
    main()