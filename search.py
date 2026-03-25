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


def ensure_lsi_index(inst, output_dir):
    """Build LSI index if not already present."""
    if not os.path.exists(os.path.join(output_dir, "lsi_index.pkl")):
        print("LSI index not found — building now ...")
        inst.build_lsi_index()
        print("Done.\n")


def ensure_positional_index(inst, output_dir):
    """Build positional index if not already present."""
    pos_file = os.path.join(output_dir, "positional_index.index")
    if not os.path.exists(pos_file):
        print("Positional index not found — building now ...")
        inst.build_positional_index()
        print("Done.\n")


def run_queries(inst, queries, method, k):
    retrieve = {
        'tfidf':    inst.retrieve_tfidf,
        'bm25':     inst.retrieve_bm25,
        'wand':     inst.retrieve_bm25_wand,
        'prf':      inst.retrieve_bm25_prf,
        'phrase':   inst.retrieve_phrase,
        'lsi':      inst.retrieve_lsi,
        'prefix':   inst.retrieve_prefix,
        'wildcard': inst.retrieve_wildcard,
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
    parser.add_argument("--method", choices=["tfidf", "bm25", "wand", "prf", "phrase", "lsi",
                                               "prefix", "wildcard"], default="bm25",
                        help="Retrieval method (default: bm25)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of results to return (default: 10)")
    args = parser.parse_args()

    output_dir = "index_spimi" if args.indexer == "spimi" else "index"
    os.makedirs(output_dir, exist_ok=True)

    inst = build_index(args.indexer, output_dir)
    if args.method == "phrase":
        ensure_positional_index(inst, output_dir)
    if args.method == "lsi":
        ensure_lsi_index(inst, output_dir)
    run_queries(inst, QUERIES, args.method, args.k)
