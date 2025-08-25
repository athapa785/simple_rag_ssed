

#!/usr/bin/env python3
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import argparse
from rag_simple.config import Config
from rag_simple.generate import answer


def main():
    p = argparse.ArgumentParser(description="Ask a question against the RAG index")
    p.add_argument("question", help="Your question")
    args = p.parse_args()

    cfg = Config()
    resp = answer(cfg, args.question)

    print("\n=== ANSWER ===\n")
    print(resp["answer"]) 
    print("\n=== SOURCES ===\n")
    for s in resp.get("sources", []):
        src = s.get("source")
        page = s.get("page")
        chunk = s.get("chunk")
        score = s.get("score")
        print(f"- {src} (page {page}, chunk {chunk}, dist {score:.4f})")


if __name__ == "__main__":
    main()