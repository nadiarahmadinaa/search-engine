# Search Engine from Scratch

A ranked information retrieval system built on an inverted index. Implements two indexing strategies (BSBI and SPIMI), five scoring/retrieval methods, bit-level postings compression, WAND pruning, and five bonus IR features.

**Corpus:** 1,033 MEDLINE biomedical abstracts across 11 subdirectories. Relevance judgments (`qrels.txt`) and 30 TREC queries are included for evaluation.

---

## Requirements

```bash
pip install nltk tqdm numpy scipy scikit-learn
python -c "import nltk; nltk.download('stopwords')"
```

---

## Quick Start

```bash
# 1. Build the SPIMI index (default)
python3 spimi.py

# 2. Search
python3 search.py --query "lipid metabolism toxemia pregnancy" --method bm25 --k 10

# 3. Evaluate all methods on 30 queries
python3 evaluation.py
```

---

## Indexing

Two indexers are available, producing logically identical merged inverted indices:

| Indexer | Script | Term keys | Block trigger |
|---------|--------|-----------|---------------|
| **SPIMI** (default) | `spimi.py` | string (no global ID map) | memory threshold (50K unique pairs/block) |
| **BSBI** | `bsbi.py` | integer termID | directory boundary (1 block per subdirectory) |

Both apply the same preprocessing pipeline: regex tokenisation → English stopword removal (NLTK) → Porter stemming.

```bash
python3 spimi.py   # builds index_spimi/
python3 bsbi.py    # builds index/
```

---

## Retrieval Methods

All methods are available through `search.py` via `--method` and through `BSBIIndex`/`SPIMIIndex` directly.

```bash
python3 search.py --query "lipid metabolism" --method <method> --indexer <bsbi|spimi> --k 10
```

| `--method` | Description |
|------------|-------------|
| `tfidf` | Log-normalised TF × IDF, no length norm |
| `bm25` | Okapi BM25 (k1=1.2, b=0.75) |
| `wand` | BM25 with WAND top-K pruning — exact same results, skips provably non-competitive docs |
| `prf` | BM25 + Pseudo-Relevance Feedback (Rocchio) — expands query with top-10 feedback terms |
| `phrase` | Positional phrase query — exact consecutive-token match, ranked by occurrence count |
| `lsi` | Latent Semantic Indexing (TruncatedSVD, 100 components) — cosine similarity in latent space |
| `prefix` | BM25 with prefix expansion via Patricia Trie (e.g. `"lipi"` → lipid, lipids, …) |
| `wildcard` | BM25 with fnmatch wildcard expansion via Patricia Trie (e.g. `"meta*"`, `"inf?mm*"`) |

### API usage

```python
from spimi import SPIMIIndex
from postings_encoding import VBEPostings

idx = SPIMIIndex(data_dir='collection', output_dir='index_spimi',
                 postings_encoding=VBEPostings)

idx.retrieve_bm25("lipid metabolism toxemia pregnancy", k=10)
idx.retrieve_bm25_prf("lipid metabolism", k=10)
idx.retrieve_phrase("blood pressure", k=10)
idx.retrieve_lsi("lipid metabolism", k=10)
idx.retrieve_prefix("lipi metab", k=10)   # prefix expansion
idx.retrieve_wildcard("lipid* metabo*", k=10)  # wildcard expansion
```

All methods return `List[Tuple[float, str]]` — `(score, doc_path)` sorted by descending score.

---

## Evaluation

```bash
python3 evaluation.py                  # both BSBI and SPIMI, all methods
python3 evaluation.py --indexer spimi  # SPIMI only
python3 evaluation.py --indexer bsbi   # BSBI only
```

Metrics computed over top-1000 results for all 30 queries:

| Method | RBP | DCG | NDCG | AP |
|--------|-----|-----|------|----|
| TF-IDF | 0.6651 | 5.8632 | 0.7724 | 0.5004 |
| BM25 | 0.6930 | 6.0133 | 0.7923 | 0.5346 |
| BM25 + PRF | 0.7083 | 6.3131 | 0.8177 | 0.5920 |
| **LSI** | **0.7798** | **6.8542** | **0.8904** | **0.7034** |

LSI achieves the best results on every metric. PRF delivers +10.7% AP over BM25. WAND returns identical results to BM25 with fewer full-score computations.

---

## Bonus Features

### 1. SPIMI Indexing (`spimi.py`)
Single-Pass In-Memory Indexing. Builds blocks using string terms instead of a global termID map, flushing to disk when the in-memory unique (term, doc) count exceeds a threshold. Scales to large corpora without the global ID-map bottleneck. Produces identical retrieval quality to BSBI.

### 2. Pseudo-Relevance Feedback / Rocchio (`bsbi.py: retrieve_bm25_prf`)
Two-phase retrieval: (1) initial BM25 run, (2) centroid of top-10 feedback docs computed in TF-IDF space, (3) Rocchio update expands the query with top-10 new terms, (4) BM25 re-retrieval on expanded query. Handles BSBI integer term keys and SPIMI string term keys transparently via `_str_to_term`/`_term_to_str` polymorphism.

**Gain over BM25:** RBP +2.2%, DCG +5.0%, NDCG +3.2%, AP +10.7%

### 3. Positional Index + Phrase Queries (`positional_index.py`, `bsbi.py: retrieve_phrase`)
Extends the postings format to store per-document token positions (VBE gap-encoded). `positional_intersect()` checks consecutive-position constraints across terms. Multi-word phrase queries are evaluated pairwise left-to-right. Results are ranked by phrase occurrence count.

```bash
python3 search.py --query "blood pressure" --method phrase
python3 bsbi.py  # then:
python3 bsbi.py --build-positional  # or auto-built on first phrase query
```

### 4. Latent Semantic Indexing (`lsi.py`, `bsbi.py: retrieve_lsi`)
Builds a sparse log-TF·IDF matrix (1033 × 8894), decomposes it with TruncatedSVD (100 components), and L2-normalises all doc vectors. Retrieval projects the query into the latent space and ranks by cosine similarity. Captures semantic relatedness beyond exact vocabulary overlap. Model persisted to `lsi_index.pkl`.

```bash
python3 search.py --query "heart disease" --method lsi
```

**Gain over BM25:** RBP +12.5%, DCG +14.0%, NDCG +12.4%, AP +31.6%

### 5. Patricia Trie Term Dictionary (`patricia_trie.py`)
A compressed radix trie over the vocabulary. Enables operations impossible with a plain `dict`:
- **Prefix search** — `get_terms_by_prefix("lipi")` → all terms starting with "lipi" in O(|prefix| + output)
- **Wildcard search** — `wildcard_search("meta*")` → 19 matching terms via fnmatch

The trie is built lazily on first use and cached on the index instance. `BSBIIndex` builds it from `term_id_map`; `SPIMIIndex` reads string terms directly from the merged index.

```bash
python3 search.py --query "lipi* metabo*" --method wildcard
python3 search.py --query "lipi metab"    --method prefix
```

---

## Compression

Two postings encodings in `postings_encoding.py `:

| Class | Scheme | Best for |
|-------|--------|---------|
| `VBEPostings` | Variable-Byte Encoding — 7 bits/byte payload, continuation bit | General use (default) |
| `EliasGammaPostings` | Bit-level Elias-Gamma — `k` zeros + stop bit + `k`-bit suffix | Small gaps (high-frequency terms) |

Pass either to any index constructor via `postings_encoding=`.

---

## File Structure

```
.
├── collection/            # Corpus (1,033 docs, 11 subdirs)
├── index/                 # BSBI index output
├── index_spimi/           # SPIMI index output
├── bsbi.py                # BSBI indexer + all retrieval methods
├── spimi.py               # SPIMI indexer (subclasses BSBIIndex)
├── postings_encoding.py   # VBEPostings, EliasGammaPostings
├── evaluation.py          # RBP, DCG, NDCG, AP evaluation
├── index.py               # InvertedIndexReader / Writer + Patricia Trie integration
├── positional_index.py    # PositionalIndexReader / Writer + positional_intersect
├── lsi.py                 # LSIIndex (TruncatedSVD build, retrieve, save, load)
├── patricia_trie.py       # PatriciaTrie (insert, prefix search, wildcard search)
├── search.py              # CLI search interface
├── util.py                # IdMap, helper utilities
├── qrels.txt              # Relevance judgments (TREC format)
└── queries.txt            # 30 evaluation queries
```

---

## References

- Manning, Raghavan & Schütze — *Introduction to Information Retrieval* (2008)
- Robertson et al. — *Okapi at TREC-3* (1994) — BM25
- Broder et al. — *Efficient query evaluation using a two-level retrieval process* (2003) — WAND
- Buckley & Salton — *Optimization of relevance feedback weights* (1995) — Rocchio/PRF
- Deerwester et al. — *Indexing by Latent Semantic Analysis* (1990) — LSI
- Morrison — *PATRICIA* (1968) — Patricia Trie
- Elias — *Universal codeword sets and representations of the integers* (1975) — Elias-Gamma
- Järvelin & Kekäläinen — *Cumulated gain-based evaluation of IR techniques* (2002) — NDCG
- Moffat & Zobel — *Rank-biased precision for measurement of retrieval effectiveness* (2008) — RBP
