"""
search.py — interactive retrieval demo.

Usage
-----
    # BSBI
    python3 search.py
    python3 search.py --indexer bsbi --method bm25 --k 10

    # SPIMI (default)
    python3 search.py --indexer spimi
    python3 search.py --indexer spimi --method tfidf --k 5
"""

import argparse
import os

from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings

QUERIES = [
    "alkylated with radioactive iodoacetate",
    "psychodrama for disturbed children",
    "lipid metabolism in toxemia and normal pregnancy",
]


def build_index(indexer, output_dir):
    """Build index only if it does not already exist."""
    index_file = os.path.join(output_dir, "main_index.index")
    if indexer == "spimi":
        index_file = os.path.join(output_dir, "main_index_spimi.index")
        inst = SPIMIIndex(data_dir='collection',
                          output_dir=output_dir,
                          postings_encoding=VBEPostings)
    else:
        inst = BSBIIndex(data_dir='collection',
                         output_dir=output_dir,
                         postings_encoding=VBEPostings)

    if not os.path.exists(index_file):
        print(f"[{indexer.upper()}] Index not found — building now ...")
        inst.index()
        print("Done.\n")
    return inst


def run_queries(inst, queries, method, k):
    retrieve = {
        'tfidf': inst.retrieve_tfidf,
        'bm25':  inst.retrieve_bm25,
        'wand':  inst.retrieve_bm25_wand,
        'prf':   inst.retrieve_bm25_prf,
    }[method]

    for query in queries:
        print(f"Query  : {query}")
        print(f"Results ({method.upper()}, k={k}):")
        for score, doc in retrieve(query, k=k):
            print(f"  {doc:50s}  {score:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexer", choices=["bsbi", "spimi"], default="spimi",
                        help="Which indexer to use (default: spimi)")
    parser.add_argument("--method", choices=["tfidf", "bm25", "wand", "prf"], default="bm25",
                        help="Retrieval method (default: bm25)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of results to return (default: 10)")
    args = parser.parse_args()

    output_dir = "index_spimi" if args.indexer == "spimi" else "index"
    os.makedirs(output_dir, exist_ok=True)

    inst = build_index(args.indexer, output_dir)
    run_queries(inst, QUERIES, args.method, args.k)
