import os
import re
import pickle
import contextlib
import heapq
import time
import math
from bisect import bisect_left

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from postings_encoding import StandardPostings, VBEPostings, EliasGammaPostings
from positional_index import (PositionalIndexWriter, PositionalIndexReader,
                               positional_intersect)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STOPWORDS = set(stopwords.words('english'))
_STEMMER   = PorterStemmer()


def preprocess(text):
    """
    Tokenize, lowercase, remove stopwords, and stem text using NLTK's
    Porter Stemmer and English stopword list.

    Parameters
    ----------
    text : str
        Raw document or query text.

    Returns
    -------
    List[str]
        Preprocessed tokens ready for indexing or lookup.
    """
    tokens = re.findall(r'[a-z]+', text.lower())
    return [_STEMMER.stem(t) for t in tokens if t not in _STOPWORDS]

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in preprocess(f.read()):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            scores = {}
            seen = set()
            for word in preprocess(query):
                if word in seen:
                    continue
                seen.add(word)
                if word not in self.term_id_map.str_to_id:
                    continue
                term = self.term_id_map[word]
                if term not in merged_index.postings_dict:
                    continue
                df = merged_index.postings_dict[term][1]
                postings, tf_list = merged_index.get_postings_list(term)
                for doc_id, tf in zip(postings, tf_list):
                    if tf > 0:
                        scores[doc_id] = scores.get(doc_id, 0) + \
                            math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for doc_id, score in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Ranked retrieval using Okapi BM25 (Robertson et al., 1994).

        Scoring is TaaT (Term-at-a-Time). For each query term t:

            IDF(t)    = log((N - df + 0.5) / (df + 0.5) + 1)

            tf_norm   = tf(t,D) * (k1 + 1)
                        ─────────────────────────────────────────
                        tf(t,D) + k1 * (1 - b + b * |D| / avgdl)

            Score(D)  += IDF(t) * tf_norm

        Document lengths (|D|) are precomputed at index time and stored in
        merged_index.doc_length.  avgdl is derived from those values.

        Parameters
        ----------
        query : str
            Raw query string (same preprocessing as indexing is applied).
        k : int
            Number of top results to return.
        k1 : float
            TF saturation parameter (typical range 1.2 – 2.0).
        b : float
            Length normalisation parameter (0 = none, 1 = full).

        Returns
        -------
        List[Tuple[float, str]]
            Top-k (score, doc_path) pairs sorted by descending score.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N     = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            seen  = set()
            for word in preprocess(query):
                if word in seen:
                    continue
                seen.add(word)
                if word not in self.term_id_map.str_to_id:
                    continue
                term = self.term_id_map[word]
                if term not in merged_index.postings_dict:
                    continue

                df  = merged_index.postings_dict[term][1]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                postings, tf_list = merged_index.get_postings_list(term)

                for doc_id, tf in zip(postings, tf_list):
                    dl      = merged_index.doc_length[doc_id]
                    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                    scores[doc_id] = scores.get(doc_id, 0) + idf * tf_norm

        docs = [(score, self.doc_id_map[doc_id]) for doc_id, score in scores.items()]
        return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def _store_upper_bounds(self, k1=1.2, b=0.75):
        """
        Compute and persist the BM25 upper bound for every term.

        The upper bound for term t is:

            UB(t) = IDF(t) * max over all D in postings(t) of tf_norm(t, D)

        Iterates the merged index sequentially (one disk pass) and writes
        the resulting upper_bounds dict back into the index metadata via
        the context manager's __exit__ save.  Must be called after index()
        has built the merged index.

        Parameters
        ----------
        k1 : float
            BM25 k1 parameter (must match the value used in retrieval).
        b : float
            BM25 b parameter (must match the value used in retrieval).
        """
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N     = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N

            for term, postings, tf_list in tqdm(merged_index,
                                                desc="Computing WAND upper bounds"):
                df  = merged_index.postings_dict[term][1]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                max_tf_norm = max(
                    (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * merged_index.doc_length[doc_id] / avgdl))
                    for doc_id, tf in zip(postings, tf_list)
                )
                merged_index.term_upper_bounds[term] = idf * max_tf_norm
            # __exit__ saves term_upper_bounds into the .dict metadata file

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        WAND (Weak AND) Top-K retrieval with BM25 scoring.

        Uses precomputed per-term upper bounds stored in the index to
        prune documents that cannot possibly enter the top-K heap, avoiding
        a full BM25 evaluation for every document.

        Algorithm (Broder et al., 2003):
          1. Sort query-term cursors by their current docID.
          2. Find the pivot: the first term where the cumulative UB sum
             exceeds the current threshold (the K-th best score so far).
          3. If the first cursor is already at pivot_doc → full eval:
               compute true BM25 for pivot_doc, update heap and threshold.
          4. Otherwise → skip: advance the first cursor to pivot_doc using
               binary search (O(log n) per skip).
          5. Repeat until no pivot can beat the threshold.

        Parameters
        ----------
        query : str
            Raw query string.
        k : int
            Number of top results to return.
        k1 : float
            BM25 TF saturation parameter.
        b : float
            BM25 length normalisation parameter.

        Returns
        -------
        List[Tuple[float, str]]
            Top-k (score, doc_path) pairs sorted by descending score.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N     = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N

            # Collect unique in-vocabulary query terms
            seen = set()
            query_terms = []
            for word in preprocess(query):
                if word in self.term_id_map.str_to_id:
                    term = self.term_id_map[word]
                    if term in merged_index.postings_dict and term not in seen:
                        query_terms.append(term)
                        seen.add(term)

            if not query_terms:
                return []

            # Load all postings into memory; initialise cursors
            postings_data = {}          # term -> (postings_list, tf_list)
            cursors       = {}          # term -> current position (int)
            upper_bounds  = {}          # term -> UB float
            for term in query_terms:
                postings_data[term] = merged_index.get_postings_list(term)
                cursors[term]       = 0
                upper_bounds[term]  = merged_index.term_upper_bounds.get(term, float('inf'))

            # Min-heap of size ≤ k  →  heap[0] is the weakest score kept
            heap      = []   # entries: (score, doc_id)
            threshold = 0.0

            while True:
                # Active terms: those with remaining postings
                active = [t for t in query_terms
                          if cursors[t] < len(postings_data[t][0])]
                if not active:
                    break

                # Sort active terms by their current docID (ascending)
                active.sort(key=lambda t: postings_data[t][0][cursors[t]])

                # Find pivot: first term where cumulative UB > threshold
                cum_ub     = 0.0
                pivot_idx  = -1
                for i, t in enumerate(active):
                    cum_ub += upper_bounds[t]
                    if cum_ub > threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    break   # no document can beat the threshold

                pivot_doc = postings_data[active[pivot_idx]][0][cursors[active[pivot_idx]]]

                if postings_data[active[0]][0][cursors[active[0]]] == pivot_doc:
                    # All terms 0..pivot_idx are at pivot_doc → full evaluation
                    score = 0.0
                    for t in active:
                        pos = cursors[t]
                        pl, tfl = postings_data[t]
                        if pos < len(pl) and pl[pos] == pivot_doc:
                            tf      = tfl[pos]
                            dl      = merged_index.doc_length[pivot_doc]
                            df      = merged_index.postings_dict[t][1]
                            idf     = math.log((N - df + 0.5) / (df + 0.5) + 1)
                            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                            score  += idf * tf_norm
                            cursors[t] = pos + 1   # advance past pivot_doc

                    # Update top-K heap
                    if len(heap) < k:
                        heapq.heappush(heap, (score, pivot_doc))
                        if len(heap) == k:
                            threshold = heap[0][0]
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, pivot_doc))
                        threshold = heap[0][0]

                else:
                    # First cursor is before pivot_doc → skip forward with bisect
                    first = active[0]
                    pl    = postings_data[first][0]
                    cursors[first] = bisect_left(pl, pivot_doc, cursors[first])

        docs = [(score, self.doc_id_map[doc_id]) for score, doc_id in heap]
        return sorted(docs, key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # PRF helpers — overridden by SPIMIIndex for string-key indices
    # ------------------------------------------------------------------

    def _str_to_term(self, word):
        """
        Convert a preprocessed string token to the term key used in
        postings_dict.  For BSBI this is an integer termID; SPIMIIndex
        overrides this to return the string directly.

        Returns None if the word is not in the vocabulary.
        """
        return self.term_id_map.str_to_id.get(word)

    def _term_to_str(self, term):
        """
        Convert a term key (integer termID in BSBI) back to its string
        form.  SPIMIIndex overrides this to be the identity function.
        """
        return self.term_id_map[term]   # int -> str via id_to_str

    # ------------------------------------------------------------------
    # Pseudo-Relevance Feedback (Rocchio)
    # ------------------------------------------------------------------

    def retrieve_bm25_prf(self, query, k=10, top_fb=10, n_terms=10,
                          alpha=1.0, beta=0.75):
        """
        BM25 retrieval with Pseudo-Relevance Feedback (Rocchio).

        Two-phase retrieval:
          Phase 1 — retrieve top_fb documents with BM25 (assumed relevant).
          Phase 2 — expand the query using Rocchio's formula, re-retrieve.

        Rocchio (no negative feedback term):
            q_new = alpha * q_orig
                  + beta * (1/|R|) * sum_{d in R} tfidf(d)

        The top n_terms dimensions of q_new become the expanded query.

        Parameters
        ----------
        query : str
            Raw query string.
        k : int
            Final number of results to return.
        top_fb : int
            Number of top documents used as pseudo-relevant feedback.
        n_terms : int
            Number of expansion terms selected from the Rocchio vector.
        alpha : float
            Weight on original query (default 1.0).
        beta : float
            Weight on feedback centroid (default 0.75).

        Returns
        -------
        List[Tuple[float, str]]
            Top-k (score, doc_path) pairs sorted by descending score.

        References
        ----------
        Rocchio (1971); Manning et al., IIR Ch. 9.
        """
        # Phase 1: initial BM25 retrieval (also triggers map load)
        initial = self.retrieve_bm25(query, k=top_fb)
        if not initial:
            return []

        # Map retrieved doc paths → integer doc_ids
        fb_doc_ids = set()
        for _, doc_path in initial:
            did = self.doc_id_map.str_to_id.get(doc_path)
            if did is not None:
                fb_doc_ids.add(did)

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as idx:
            N = len(idx.doc_length)

            # --- query vector: {str_term: idf_weight} ---
            query_vec = {}
            for word in set(preprocess(query)):
                term = self._str_to_term(word)
                if term is not None and term in idx.postings_dict:
                    df = idx.postings_dict[term][1]
                    query_vec[word] = math.log(N / df)

            # --- feedback centroid via sequential scan ---
            # For each term, accumulate log-TF * IDF over feedback docs.
            # Sequential iteration avoids random seeks inside the loop.
            centroid = {}
            for term, postings, tf_list in idx:
                for doc_id, tf in zip(postings, tf_list):
                    if doc_id in fb_doc_ids:
                        df = idx.postings_dict[term][1]
                        idf = math.log(N / df)
                        tfidf = (1 + math.log(tf)) * idf
                        word = self._term_to_str(term)
                        centroid[word] = centroid.get(word, 0.0) + tfidf

        # Normalise centroid by number of feedback docs
        n_fb = len(fb_doc_ids)
        for w in centroid:
            centroid[w] /= n_fb

        # Rocchio combination
        all_terms = set(query_vec) | set(centroid)
        expanded = {t: alpha * query_vec.get(t, 0.0) +
                       beta  * centroid.get(t, 0.0)
                    for t in all_terms}

        top_terms = sorted(expanded, key=expanded.get, reverse=True)[:n_terms]

        # Phase 2: re-retrieve with expanded query
        return self.retrieve_bm25(' '.join(top_terms), k=k)

    # ------------------------------------------------------------------
    # Latent Semantic Index
    # ------------------------------------------------------------------

    _LSI_INDEX_NAME = "lsi_index.pkl"

    def build_lsi_index(self, n_components=100):
        """
        Build a Latent Semantic Index over the merged inverted index.

        Constructs a log-TF·IDF term-document matrix (n_docs × n_terms),
        applies TruncatedSVD to find k latent topics, and stores
        L2-normalised document vectors for cosine-similarity retrieval.

        The fitted model is pickled to <output_dir>/lsi_index.pkl so it
        survives across sessions without recomputing.

        Parameters
        ----------
        n_components : int
            Number of SVD dimensions.  Default 100; must be less than
            min(n_docs, n_terms).

        References
        ----------
        Deerwester et al. (1990) — "Indexing by latent semantic analysis".
        Manning et al., IIR Ch. 18.
        """
        from lsi import LSIIndex
        from scipy.sparse import lil_matrix
        import numpy as np

        if len(self.doc_id_map) == 0:
            self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as idx:
            N       = len(idx.doc_length)
            n_terms = len(idx.terms)

            # Map each term key → column index (order = idx.terms order)
            term_to_col = {term: col for col, term in enumerate(idx.terms)}

            # Build sparse log-TF·IDF matrix  (n_docs × n_terms)
            X = lil_matrix((N, n_terms), dtype=np.float32)

            for term, postings, tf_list in tqdm(idx,
                                                total=n_terms,
                                                desc="Building TF-IDF matrix"):
                col = term_to_col[term]
                df  = len(postings)
                idf = math.log(N / df)
                for doc_id, tf in zip(postings, tf_list):
                    X[doc_id, col] = (1 + math.log(tf)) * idf

        X = X.tocsr()

        # doc_id_list[i] = path for matrix row i  (doc IDs are 0-indexed)
        doc_id_list = [self.doc_id_map[i] for i in range(N)]

        k = min(n_components, N - 1, n_terms - 1)
        lsi = LSIIndex(n_components=k)
        lsi.build(X, term_to_col, doc_id_list)
        lsi.save(os.path.join(self.output_dir, self._LSI_INDEX_NAME))
        self._lsi = lsi

    def _get_lsi(self):
        """Load or return cached LSIIndex."""
        if getattr(self, '_lsi', None) is not None:
            return self._lsi
        from lsi import LSIIndex
        path = os.path.join(self.output_dir, self._LSI_INDEX_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "LSI index not found. Run build_lsi_index() first."
            )
        self._lsi = LSIIndex.load(path)
        return self._lsi

    def retrieve_lsi(self, query, k=10):
        """
        LSI ranked retrieval using cosine similarity in latent space.

        Builds a log-IDF query vector (TF=1 for each unique query term),
        projects it into the SVD latent space, and ranks documents by
        cosine similarity.

        Parameters
        ----------
        query : str
            Raw query string (same preprocessing as indexing).
        k : int
            Number of results to return.

        Returns
        -------
        List[Tuple[float, str]]
            (cosine_score, doc_path) sorted by descending score.
        """
        import numpy as np

        if len(self.doc_id_map) == 0:
            self.load()

        lsi = self._get_lsi()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as idx:
            N         = len(idx.doc_length)
            n_terms   = len(lsi.term_to_col)
            query_vec = np.zeros(n_terms, dtype=np.float32)

            for word in set(preprocess(query)):
                term = self._str_to_term(word)
                if term is None or term not in idx.postings_dict:
                    continue
                if term not in lsi.term_to_col:
                    continue
                col             = lsi.term_to_col[term]
                df              = idx.postings_dict[term][1]
                query_vec[col]  = math.log(N / df)   # IDF weight, TF=1

        return lsi.retrieve(query_vec, k=k)

    # ------------------------------------------------------------------
    # Positional Index
    # ------------------------------------------------------------------

    _POSITIONAL_INDEX_NAME = "positional_index"

    def build_positional_index(self):
        """
        Build a positional inverted index over the entire corpus.

        Makes a single pass through all documents, tracking the token-level
        position of each term (in the preprocessed stream, i.e. after
        stopword removal and stemming).  Terms are stored as raw strings —
        no global term-ID map is needed.

        The resulting index is written to
        <output_dir>/positional_index.{index,dict}.

        Must be called after the main index has been built (so that
        doc_id_map is populated and can be loaded).
        """
        if len(self.doc_id_map) == 0:
            self.load()

        # {str_term: {doc_id: [positions]}}
        term_pos = {}

        for block_dir in tqdm(sorted(next(os.walk(self.data_dir))[1]),
                              desc="Building positional index"):
            dir_path = "./" + self.data_dir + "/" + block_dir
            for filename in sorted(next(os.walk(dir_path))[2]):
                doc_path = dir_path + "/" + filename
                doc_id   = self.doc_id_map.str_to_id.get(doc_path)
                if doc_id is None:
                    continue  # doc not in main index (shouldn't happen)

                with open(doc_path, 'r', encoding='utf8',
                          errors='surrogateescape') as f:
                    tokens = preprocess(f.read())

                for pos, token in enumerate(tokens):
                    if token not in term_pos:
                        term_pos[token] = {}
                    if doc_id not in term_pos[token]:
                        term_pos[token][doc_id] = []
                    term_pos[token][doc_id].append(pos)

        with PositionalIndexWriter(self._POSITIONAL_INDEX_NAME, VBEPostings,
                                   directory=self.output_dir) as writer:
            for term in sorted(term_pos.keys()):
                sorted_docs    = sorted(term_pos[term].keys())
                tf_list        = [len(term_pos[term][d]) for d in sorted_docs]
                positions_list = [term_pos[term][d]      for d in sorted_docs]
                writer.append(term, sorted_docs, tf_list, positions_list)

    def retrieve_phrase(self, phrase_query, k=10):
        """
        Exact-phrase retrieval using the positional inverted index.

        Preprocesses the phrase with the same pipeline as indexing (so
        stopwords are removed and tokens are stemmed), then uses pairwise
        positional intersection to find documents where all phrase tokens
        appear consecutively in the preprocessed token stream.

        Falls back to BM25 for single-token queries (no phrase to match).

        Parameters
        ----------
        phrase_query : str
            The phrase to search for (e.g. "lipid metabolism").
        k : int
            Number of results to return, ranked by phrase-occurrence count.

        Returns
        -------
        List[Tuple[int, str]]
            (phrase_count, doc_path) pairs, sorted by descending count.
        """
        if len(self.doc_id_map) == 0:
            self.load()

        words = preprocess(phrase_query)
        if not words:
            return []
        if len(words) == 1:
            return self.retrieve_bm25(phrase_query, k=k)

        pos_idx_file = os.path.join(
            self.output_dir,
            self._POSITIONAL_INDEX_NAME + '.index'
        )
        if not os.path.exists(pos_idx_file):
            raise FileNotFoundError(
                "Positional index not found. "
                "Run build_positional_index() first."
            )

        with PositionalIndexReader(self._POSITIONAL_INDEX_NAME, VBEPostings,
                                   directory=self.output_dir) as pos_idx:
            # Bail out early if any phrase word is not indexed
            for word in words:
                if word not in pos_idx.postings_dict:
                    return []

            # Load positional postings for all phrase words
            pdata = {w: pos_idx.get_postings_list_positional(w) for w in words}

        # Seed with first word: [(doc_id, [phrase_start_positions])]
        posts0, _, pos0 = pdata[words[0]]
        current = list(zip(posts0, pos0))

        # Pairwise positional intersect for each subsequent word
        for offset, word in enumerate(words[1:], start=1):
            posts_w, _, pos_w = pdata[word]
            next_lookup = dict(zip(posts_w, pos_w))
            new_current = []
            for doc_id, phrase_starts in current:
                if doc_id not in next_lookup:
                    continue
                npos = next_lookup[doc_id]
                valid = []
                j = 0
                for ps in phrase_starts:
                    target = ps + offset
                    while j < len(npos) and npos[j] < target:
                        j += 1
                    if j < len(npos) and npos[j] == target:
                        valid.append(ps)
                if valid:
                    new_current.append((doc_id, valid))
            current = new_current
            if not current:
                return []

        results = [(len(phrase_starts), self.doc_id_map[doc_id])
                   for doc_id, phrase_starts in current]
        return sorted(results, reverse=True)[:k]

    # ------------------------------------------------------------------
    # Patricia Trie — prefix / wildcard retrieval
    # ------------------------------------------------------------------

    def _get_term_trie(self):
        """
        Return a PatriciaTrie over the string vocabulary (lazily built,
        then cached on this instance).

        For BSBI the vocabulary comes from term_id_map.str_to_id.
        SPIMIIndex overrides this to read string terms directly from
        the merged index postings_dict.
        """
        if getattr(self, '_term_trie', None) is not None:
            return self._term_trie
        from patricia_trie import PatriciaTrie
        if len(self.term_id_map) == 0:
            self.load()
        trie = PatriciaTrie()
        for word in self.term_id_map.str_to_id:
            trie.insert(word, None)
        self._term_trie = trie
        return trie

    def get_terms_by_prefix(self, prefix: str):
        """
        Return all vocabulary strings that start with *prefix*.

        Uses the Patricia Trie for O(|prefix| + output) lookup.

        Parameters
        ----------
        prefix : str
            Literal prefix to search for.

        Returns
        -------
        List[str]
            Matching vocabulary terms (stemmed/preprocessed form).
        """
        return [t for t, _ in self._get_term_trie().get_terms_by_prefix(prefix)]

    def retrieve_prefix(self, query: str, k=10):
        """
        BM25 retrieval with prefix expansion.

        Each preprocessed query token is expanded to all vocabulary terms
        sharing that prefix via the Patricia Trie, then BM25 is run on
        the expanded token set.

        Parameters
        ----------
        query : str
            Raw query string; each token is used as a prefix.
        k : int
            Number of top results to return.

        Returns
        -------
        List[Tuple[float, str]]
            Top-k (score, doc_path) pairs sorted by descending score.
        """
        trie = self._get_term_trie()
        expanded = set()
        for word in preprocess(query):
            for t, _ in trie.get_terms_by_prefix(word):
                expanded.add(t)
        if not expanded:
            return []
        return self.retrieve_bm25(' '.join(expanded), k=k)

    def retrieve_wildcard(self, pattern: str, k=10):
        """
        BM25 retrieval with wildcard expansion.

        Each space-separated token in *pattern* is treated as an fnmatch
        wildcard (* and ?) against the vocabulary via the Patricia Trie.
        Matched terms are merged and scored with BM25.

        Parameters
        ----------
        pattern : str
            Wildcard query, e.g. ``"lipi* metab*"`` or ``"inf?mm*"``.
        k : int
            Number of top results to return.

        Returns
        -------
        List[Tuple[float, str]]
            Top-k (score, doc_path) pairs sorted by descending score.
        """
        trie = self._get_term_trie()
        expanded = set()
        for token in pattern.split():
            for t, _ in trie.wildcard_search(token):
                expanded.add(t)
        if not expanded:
            return []
        return self.retrieve_bm25(' '.join(expanded), k=k)

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

        self._store_upper_bounds()


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
