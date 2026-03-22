import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings


######## >>>>> IR metrics


def rbp(ranking, p=0.8):
    """
    Rank-Biased Precision (Moffat & Zobel, 2008).

    Models a user who continues reading the result list with probability p
    after each document.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector, e.g. [1, 0, 1, 1, 0].
        ranking[i] == 1 means the document at rank i+1 is relevant.
    p : float
        Persistence parameter (default 0.8).

    Returns
    -------
    float
        RBP score in [0, 1].
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking, k=None):
    """
    Discounted Cumulative Gain (Järvelin & Kekäläinen, 2002).

    Rewards relevant documents appearing at higher ranks with a
    logarithmic discount: gain at rank i = rel(i) / log2(i + 1).

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.
    k : int, optional
        Evaluate only the top-k positions. Uses full ranking if None.

    Returns
    -------
    float
        DCG score (unbounded, higher is better).
    """
    if k is not None:
        ranking = ranking[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(ranking))


def ndcg(ranking, n_relevant, k=None):
    """
    Normalized Discounted Cumulative Gain (Järvelin & Kekäläinen, 2002).

    Divides DCG by the Ideal DCG (IDCG) — the DCG of a perfect ranking
    where all relevant documents appear first. Normalises to [0, 1] so
    scores are comparable across queries with different numbers of
    relevant documents.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.
    n_relevant : int
        Total number of relevant documents for this query in the
        collection (used to build the ideal ranking).
    k : int, optional
        Evaluate only the top-k positions.

    Returns
    -------
    float
        NDCG score in [0, 1].
    """
    if k is not None:
        ranking = ranking[:k]

    actual_dcg = dcg(ranking)

    # Ideal ranking: as many 1s as possible at the top
    ideal_len = min(n_relevant, len(ranking))
    ideal_ranking = [1] * ideal_len + [0] * (len(ranking) - ideal_len)
    ideal_dcg = dcg(ideal_ranking)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def ap(ranking, n_relevant):
    """
    Average Precision (AP).

    For each rank position where a relevant document appears, computes
    precision at that rank, then averages over all relevant documents.
    Rewards both early retrieval and high recall.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.
    n_relevant : int
        Total number of relevant documents for this query in the
        collection (used as the denominator).

    Returns
    -------
    float
        AP score in [0, 1].
    """
    if n_relevant == 0:
        return 0.0
    hits  = 0
    score = 0.0
    for i, rel in enumerate(ranking, start=1):
        if rel == 1:
            hits  += 1
            score += hits / i
    return score / n_relevant


######## >>>>> load qrels


def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """
    Load query relevance judgments from a TREC-format qrels file.

    Returns qrels[qid][doc_id] = 1 if relevant, 0 otherwise.
    """
    qrels = {"Q" + str(i): {j: 0 for j in range(1, max_doc_id + 1)}
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as f:
        for line in f:
            parts = line.strip().split()
            qrels[parts[0]][int(parts[1])] = 1
    return qrels


######## >>>>> evaluation


def eval(qrels, query_file="queries.txt", k=1000):
    """
    Evaluate TF-IDF and BM25 against all queries using four metrics:
    RBP, DCG, NDCG, and AP (MAP).

    Parameters
    ----------
    qrels : dict
        Relevance judgments from load_qrels().
    query_file : str
        Path to the queries file.
    k : int
        Number of top results to retrieve per query.
    """
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    methods = {
        'TF-IDF': BSBI_instance.retrieve_tfidf,
        'BM25':   BSBI_instance.retrieve_bm25,
    }

    for method_name, retrieve_fn in methods.items():
        rbp_scores, dcg_scores, ndcg_scores, ap_scores = [], [], [], []

        with open(query_file) as f:
            for qline in f:
                parts = qline.strip().split()
                qid   = parts[0]
                query = " ".join(parts[1:])
                n_rel = sum(qrels[qid].values())

                ranking = []
                for _, doc in retrieve_fn(query, k=k):
                    did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                    ranking.append(qrels[qid][did])

                rbp_scores.append(rbp(ranking))
                dcg_scores.append(dcg(ranking))
                ndcg_scores.append(ndcg(ranking, n_rel))
                ap_scores.append(ap(ranking, n_rel))

        print(f"\nEvaluasi {method_name} terhadap 30 queries (top-{k}):")
        print(f"  RBP  = {sum(rbp_scores)  / len(rbp_scores):.4f}")
        print(f"  DCG  = {sum(dcg_scores)  / len(dcg_scores):.4f}")
        print(f"  NDCG = {sum(ndcg_scores) / len(ndcg_scores):.4f}")
        print(f"  AP   = {sum(ap_scores)   / len(ap_scores):.4f}")


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    # Sanity-check metrics on a known ranking
    # ranking [1,0,1] with 3 relevant docs total
    assert abs(dcg([1, 0, 1]) - (1/math.log2(2) + 1/math.log2(4))) < 1e-9
    assert ndcg([1, 0, 1], n_relevant=3) < 1.0   # not perfect
    assert ndcg([1, 1, 1], n_relevant=3) == 1.0   # perfect
    assert abs(ap([1, 0, 1], n_relevant=2) - (1/1 + 2/3) / 2) < 1e-9
    print("Semua assertion metric passed.\n")

    eval(qrels)
