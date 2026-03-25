"""
SPIMI (Single-Pass In-Memory Indexing) implementation for TP2.

Key differences from the existing BSBI indexer:
  1. No global term-ID map during block construction — blocks use raw string
     terms as dictionary keys.
  2. Memory-threshold flushing — blocks are written to disk whenever the
     cumulative unique (term, doc) pair count exceeds max_postings_per_block,
     not at fixed directory boundaries.
  3. Single sequential pass over all documents across all directories.

The on-disk merged index uses string terms as postings_dict keys (instead of
integer termIDs), so retrieval methods are adapted accordingly.

References
----------
Manning, Raghavan & Schütze — IIR, Ch. 4.2 (SPIMI), Cambridge UP, 2008.
"""

import os
import sys
import contextlib
import heapq
import math
import time
from bisect import bisect_left

from tqdm import tqdm

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import VBEPostings
from bsbi import BSBIIndex, preprocess


class SPIMIIndex(BSBIIndex):
    """
    Single-Pass In-Memory Indexing.

    Subclasses BSBIIndex but overrides the indexing pipeline and all
    retrieval methods.  The merge, postings-encoding, and doc_length
    infrastructure are reused unchanged.

    Parameters
    ----------
    data_dir : str
        Path to the corpus root (same as BSBIIndex).
    output_dir : str
        Directory for index files (should differ from BSBI's output_dir).
    postings_encoding : class
        Postings codec (StandardPostings / VBEPostings / EliasGammaPostings).
    index_name : str
        Base name for the merged index files.
    max_postings_per_block : int
        Flush to disk when the in-memory unique (term, doc) pair count
        exceeds this value.  Lower values → more blocks, less peak memory.
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index_spimi",
                 max_postings_per_block=50_000):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        self.max_postings_per_block = max_postings_per_block
        self._block_counter = 0

    # ------------------------------------------------------------------
    # Block construction
    # ------------------------------------------------------------------

    def _flush_block(self, term_dict):
        """
        Sort accumulated terms alphabetically, write one block file to disk,
        clear the in-memory dictionary.

        Parameters
        ----------
        term_dict : dict[str, dict[int, int]]
            Maps string term → {doc_id: tf}.

        Returns
        -------
        str
            The block index name (without extension).
        """
        block_id = f"spimi_block_{self._block_counter}"
        self._block_counter += 1
        self.intermediate_indices.append(block_id)

        with InvertedIndexWriter(block_id, self.postings_encoding,
                                 directory=self.output_dir) as writer:
            for term in sorted(term_dict.keys()):  # alphabetic sort
                sorted_docs = sorted(term_dict[term].keys())
                tf_list = [term_dict[term][doc_id] for doc_id in sorted_docs]
                writer.append(term, sorted_docs, tf_list)

        return block_id

    # ------------------------------------------------------------------
    # Main indexing entry point
    # ------------------------------------------------------------------

    def index(self):
        """
        SPIMI index construction.

        Streams all documents in a single pass across all block directories.
        Builds an in-memory hash {str_term: {doc_id: tf}} and flushes to disk
        whenever total unique (term, doc) pairs exceeds max_postings_per_block.
        After all documents are processed, merges all block files using the
        same external merge-sort as BSBI.

        No global term→ID assignment is made during block construction —
        the merged index stores string term keys directly.
        """
        term_dict = {}       # str -> {doc_id: tf}
        total_postings = 0   # unique (term, doc) pairs accumulated so far

        all_block_dirs = sorted(next(os.walk(self.data_dir))[1])

        for block_dir in tqdm(all_block_dirs, desc="SPIMI indexing"):
            dir_path = "./" + self.data_dir + "/" + block_dir
            for filename in sorted(next(os.walk(dir_path))[2]):
                doc_path = dir_path + "/" + filename
                doc_id = self.doc_id_map[doc_path]  # auto-assigns new int ID

                with open(doc_path, 'r', encoding='utf8',
                          errors='surrogateescape') as f:
                    for token in preprocess(f.read()):
                        # SPIMI key insight: use string token directly,
                        # no conversion through a global term-ID map.
                        if token not in term_dict:
                            term_dict[token] = {}
                        if doc_id not in term_dict[token]:
                            term_dict[token][doc_id] = 0
                            total_postings += 1   # new unique pair
                        term_dict[token][doc_id] += 1

                # Check memory threshold after each document
                if total_postings >= self.max_postings_per_block:
                    self._flush_block(term_dict)
                    term_dict = {}
                    total_postings = 0

        # Flush any remaining terms
        if term_dict:
            self._flush_block(term_dict)

        # Persist doc_id_map so retrieval works across sessions.
        # term_id_map is intentionally left empty for SPIMI.
        self.save()

        # External merge of all block files into one merged index.
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(idx, self.postings_encoding,
                                           directory=self.output_dir)
                    )
                    for idx in self.intermediate_indices
                ]
                self.merge(indices, merged_index)

        # Precompute BM25 upper bounds for WAND (inherited, works with
        # string term keys transparently).
        self._store_upper_bounds()

    # ------------------------------------------------------------------
    # PRF helpers — override BSBI's integer-key versions
    # ------------------------------------------------------------------

    def _str_to_term(self, word):
        """SPIMI: term key IS the string token — identity function."""
        return word

    def _term_to_str(self, term):
        """SPIMI: term is already a string — identity function."""
        return term

    # ------------------------------------------------------------------
    # Patricia Trie — override for string-keyed SPIMI index
    # ------------------------------------------------------------------

    def _get_term_trie(self):
        """
        Build a PatriciaTrie from the SPIMI merged index string terms.

        Unlike BSBI, SPIMI has no term_id_map — the vocabulary comes
        directly from the merged index postings_dict keys (already strings).
        """
        if getattr(self, '_term_trie', None) is not None:
            return self._term_trie
        from patricia_trie import PatriciaTrie
        self._load_if_needed()
        trie = PatriciaTrie()
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as idx:
            for term in idx.postings_dict:
                trie.insert(term, None)
        self._term_trie = trie
        return trie

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _load_if_needed(self):
        """Load doc_id_map from disk when it is empty (new session)."""
        if len(self.doc_id_map) == 0:
            self.load()

    # ------------------------------------------------------------------
    # TF-IDF retrieval
    # ------------------------------------------------------------------

    def retrieve_tfidf(self, query, k=10):
        """
        TF-IDF ranked retrieval adapted for string term keys.

        Identical scoring formula to BSBIIndex.retrieve_tfidf but bypasses
        the term_id_map lookup — query tokens are used directly as
        postings_dict keys.
        """
        self._load_if_needed()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            scores = {}
            seen = set()

            for term in preprocess(query):
                if term in seen:
                    continue
                seen.add(term)
                if term not in merged_index.postings_dict:
                    continue
                df = merged_index.postings_dict[term][1]
                postings, tf_list = merged_index.get_postings_list(term)
                for doc_id, tf in zip(postings, tf_list):
                    if tf > 0:
                        scores[doc_id] = scores.get(doc_id, 0) + \
                            math.log(N / df) * (1 + math.log(tf))

        docs = [(score, self.doc_id_map[doc_id])
                for doc_id, score in scores.items()]
        return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        BM25 ranked retrieval adapted for string term keys.

        Same Okapi BM25 formula as BSBIIndex.retrieve_bm25.
        """
        self._load_if_needed()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N
            scores = {}
            seen = set()

            for term in preprocess(query):
                if term in seen:
                    continue
                seen.add(term)
                if term not in merged_index.postings_dict:
                    continue
                df = merged_index.postings_dict[term][1]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                postings, tf_list = merged_index.get_postings_list(term)
                for doc_id, tf in zip(postings, tf_list):
                    dl = merged_index.doc_length[doc_id]
                    tf_norm = (tf * (k1 + 1)) / \
                        (tf + k1 * (1 - b + b * dl / avgdl))
                    scores[doc_id] = scores.get(doc_id, 0) + idf * tf_norm

        docs = [(score, self.doc_id_map[doc_id])
                for doc_id, score in scores.items()]
        return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    # ------------------------------------------------------------------
    # WAND retrieval
    # ------------------------------------------------------------------

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        WAND top-K retrieval adapted for string term keys.

        Same algorithm as BSBIIndex.retrieve_bm25_wand but query term lookup
        goes directly to postings_dict without the term_id_map indirection.
        """
        self._load_if_needed()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N

            # Collect unique in-vocabulary query terms (as strings)
            seen = set()
            query_terms = []
            for word in preprocess(query):
                if word in merged_index.postings_dict and word not in seen:
                    query_terms.append(word)
                    seen.add(word)

            if not query_terms:
                return []

            postings_data = {}
            cursors = {}
            upper_bounds = {}
            for term in query_terms:
                postings_data[term] = merged_index.get_postings_list(term)
                cursors[term] = 0
                upper_bounds[term] = merged_index.term_upper_bounds.get(
                    term, float('inf'))

            heap = []
            threshold = 0.0

            while True:
                active = [t for t in query_terms
                          if cursors[t] < len(postings_data[t][0])]
                if not active:
                    break

                active.sort(key=lambda t: postings_data[t][0][cursors[t]])

                cum_ub = 0.0
                pivot_idx = -1
                for i, t in enumerate(active):
                    cum_ub += upper_bounds[t]
                    if cum_ub > threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    break

                pivot_doc = postings_data[active[pivot_idx]][0][
                    cursors[active[pivot_idx]]]

                if postings_data[active[0]][0][cursors[active[0]]] == pivot_doc:
                    score = 0.0
                    for t in active:
                        pos = cursors[t]
                        pl, tfl = postings_data[t]
                        if pos < len(pl) and pl[pos] == pivot_doc:
                            tf = tfl[pos]
                            dl = merged_index.doc_length[pivot_doc]
                            df = merged_index.postings_dict[t][1]
                            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                            tf_norm = (tf * (k1 + 1)) / \
                                (tf + k1 * (1 - b + b * dl / avgdl))
                            score += idf * tf_norm
                            cursors[t] = pos + 1
                    if len(heap) < k:
                        heapq.heappush(heap, (score, pivot_doc))
                        if len(heap) == k:
                            threshold = heap[0][0]
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, pivot_doc))
                        threshold = heap[0][0]
                else:
                    first = active[0]
                    pl = postings_data[first][0]
                    cursors[first] = bisect_left(pl, pivot_doc, cursors[first])

        docs = [(score, self.doc_id_map[doc_id]) for score, doc_id in heap]
        return sorted(docs, key=lambda x: x[0], reverse=True)


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil

    output_dir = "index_spimi"
    os.makedirs(output_dir, exist_ok=True)

    spimi = SPIMIIndex(
        data_dir='collection',
        output_dir=output_dir,
        postings_encoding=VBEPostings,
        max_postings_per_block=50_000,
    )

    print("Building SPIMI index ...")
    t0 = time.time()
    spimi.index()
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s  |  {spimi._block_counter} blocks flushed")

    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]

    for q in queries:
        print(f"\nQuery : {q}")
        print("BM25  :")
        for score, doc in spimi.retrieve_bm25(q, k=5):
            print(f"  {doc:50s}  {score:.4f}")
