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
from compression import StandardPostings, VBEPostings, EliasGammaPostings
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
