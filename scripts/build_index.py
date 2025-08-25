

#!/usr/bin/env python3
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