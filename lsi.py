"""
Latent Semantic Indexing (LSI / LSA) for TP2.

Builds a log-TF·IDF term-document matrix, applies Truncated SVD to
project into a k-dimensional latent semantic space, and retrieves
documents by cosine similarity in that space.

The key advantage over BM25: terms with related meanings (synonyms,
co-occurring context words) end up close in the latent space, so a
query about "myocardial infarction" can retrieve docs that say "heart
attack" without any term overlap.

Algorithm
---------
1.  Build sparse TF-IDF matrix X  (n_docs × n_terms)
2.  Truncated SVD:  X ≈ U · Σ · Vᵀ   (k components)
3.  Document vectors:  D = U · Σ  (n_docs × k), L2-normalised
4.  Query projection:  q_lsi = Vᵀ · q_tfidf  (via svd.transform)
5.  Cosine similarity: scores = D · q_lsi_norm

References
----------
Deerwester et al. (1990) — "Indexing by latent semantic analysis",
    JASIS 41(6): 391–407.
Manning, Raghavan & Schütze — IIR, Ch. 18.
"""

import pickle
import math

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class LSIIndex:
    """
    Latent Semantic Index.

    Parameters
    ----------
    n_components : int
        Number of SVD dimensions (latent topics).  Typical values: 100–300.
        Must be < min(n_docs, n_terms).
    """

    def __init__(self, n_components=100):
        self.n_components = n_components
        self.svd          = TruncatedSVD(n_components=n_components,
                                         random_state=42)
        self.term_to_col  = {}     # term key  →  column index in X
        self.doc_vectors  = None   # shape (n_docs, k), L2-normalised rows
        self.doc_id_list  = []     # doc_id_list[i] = doc path string for row i

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, X, term_to_col, doc_id_list):
        """
        Fit TruncatedSVD on X and store normalised document vectors.

        Parameters
        ----------
        X : scipy.sparse matrix, shape (n_docs, n_terms)
            Log-TF · IDF weighted term-document matrix.
        term_to_col : dict
            Maps each term key (int for BSBI, str for SPIMI) to its
            column index in X.
        doc_id_list : list[str]
            doc_id_list[i] = document path string for row i of X.
        """
        self.term_to_col = term_to_col
        self.doc_id_list = doc_id_list

        raw = self.svd.fit_transform(X)          # (n_docs, k)
        self.doc_vectors = normalize(raw, norm='l2')

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query_vec, k=10):
        """
        Project query into LSI space, rank documents by cosine similarity.

        Parameters
        ----------
        query_vec : np.ndarray, shape (n_terms,)
            Sparse TF-IDF vector for the query (0 for unknown terms).
        k : int
            Number of top results to return.

        Returns
        -------
        List[Tuple[float, str]]
            (cosine_score, doc_path) sorted by descending score.
        """
        if np.all(query_vec == 0):
            return []

        q_lsi  = self.svd.transform(query_vec.reshape(1, -1))  # (1, k)
        q_lsi  = normalize(q_lsi, norm='l2')
        scores = (self.doc_vectors @ q_lsi.T).squeeze()         # (n_docs,)

        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(float(scores[i]), self.doc_id_list[i]) for i in top_idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Pickle the fitted LSI model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'n_components': self.n_components,
                'svd':          self.svd,
                'doc_vectors':  self.doc_vectors,
                'term_to_col':  self.term_to_col,
                'doc_id_list':  self.doc_id_list,
            }, f)

    @classmethod
    def load(cls, path):
        """Load a previously saved LSI model from disk."""
        with open(path, 'rb') as f:
            d = pickle.load(f)
        inst             = cls(n_components=d['n_components'])
        inst.svd         = d['svd']
        inst.doc_vectors = d['doc_vectors']
        inst.term_to_col = d['term_to_col']
        inst.doc_id_list = d['doc_id_list']
        return inst
