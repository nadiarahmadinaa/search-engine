"""
Positional Inverted Index for TP2 — supports phrase query retrieval.

On-disk format
--------------
For each term the .index file stores three contiguous byte regions:

    [enc_postings][enc_tf][enc_positions]

postings_dict[term] = (byte_offset, num_docs,
                       len_postings_bytes, len_tf_bytes, len_positions_bytes)

Position encoding
-----------------
Positions within each doc are gap-encoded and VBE-compressed independently.
All docs' compressed gaps are concatenated into a single byte string.
The tf_list tells us how many positions belong to each doc, so decoding
can split the flat number stream back into per-doc lists.

References
----------
Manning, Raghavan & Schütze — IIR, Ch. 2.4 (Positional indexes).
Zobel & Moffat (2006) — "Inverted files for text search engines", ACM Surveys.
"""

import os
import pickle

from postings_encoding import VBEPostings
from index import InvertedIndex


# ---------------------------------------------------------------------------
# Low-level position codec (always uses VBE, independent of main encoding)
# ---------------------------------------------------------------------------

def _encode_positions(positions_list):
    """
    Encode a list-of-lists of token positions into bytes.

    For each doc's position list the positions are gap-encoded, then all
    gaps across every doc are VBE-compressed into a single byte string.

    Parameters
    ----------
    positions_list : List[List[int]]
        positions_list[i] = sorted token-offset list for doc i.

    Returns
    -------
    bytes
    """
    all_gaps = []
    for pos_list in positions_list:
        if not pos_list:
            continue
        all_gaps.append(pos_list[0])            # first position (gap from 0)
        for i in range(1, len(pos_list)):
            all_gaps.append(pos_list[i] - pos_list[i - 1])
    return VBEPostings.vb_encode(all_gaps)


def _decode_positions(pos_bytes, tf_list):
    """
    Reconstruct per-doc position lists from the flat VBE byte string.

    Parameters
    ----------
    pos_bytes : bytes
        Output of _encode_positions.
    tf_list : List[int]
        tf_list[i] = number of positions stored for doc i.

    Returns
    -------
    List[List[int]]
        positions_list[i] = sorted position list for doc i.
    """
    all_gaps = VBEPostings.vb_decode(pos_bytes)
    positions_list = []
    offset = 0
    for tf in tf_list:
        doc_positions = []
        prev = 0
        for _ in range(tf):
            prev += all_gaps[offset]
            doc_positions.append(prev)
            offset += 1
        positions_list.append(doc_positions)
    return positions_list


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class PositionalIndexWriter(InvertedIndex):
    """Write-only positional inverted index."""

    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list, positions_list):
        """
        Append one term's positional postings to the index file.

        Parameters
        ----------
        term : hashable
            String term key.
        postings_list : List[int]
            Sorted doc IDs.
        tf_list : List[int]
            tf_list[i] = len(positions_list[i]).
        positions_list : List[List[int]]
            positions_list[i] = sorted token offsets in doc postings_list[i].
        """
        self.terms.append(term)

        for doc_id, tf in zip(postings_list, tf_list):
            self.doc_length[doc_id] = self.doc_length.get(doc_id, 0) + tf

        self.index_file.seek(0, os.SEEK_END)
        curr_pos = self.index_file.tell()

        enc_postings  = VBEPostings.encode(postings_list)
        enc_tf        = VBEPostings.encode_tf(tf_list)
        enc_positions = _encode_positions(positions_list)

        self.index_file.write(enc_postings)
        self.index_file.write(enc_tf)
        self.index_file.write(enc_positions)

        self.postings_dict[term] = (
            curr_pos,
            len(postings_list),
            len(enc_postings),
            len(enc_tf),
            len(enc_positions),
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class PositionalIndexReader(InvertedIndex):
    """Read-only positional inverted index."""

    def __iter__(self):
        return self

    def reset(self):
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()

    def __next__(self):
        """Yield (term, postings_list, tf_list, positions_list)."""
        curr_term = next(self.term_iter)
        _, _, len_posts, len_tf, len_pos = self.postings_dict[curr_term]
        postings  = VBEPostings.decode(self.index_file.read(len_posts))
        tf_list   = VBEPostings.decode_tf(self.index_file.read(len_tf))
        positions = _decode_positions(self.index_file.read(len_pos), tf_list)
        return curr_term, postings, tf_list, positions

    def get_postings_list_positional(self, term):
        """
        Random-access fetch of positional postings for one term.

        Returns
        -------
        Tuple[List[int], List[int], List[List[int]]]
            (postings_list, tf_list, positions_list)
        """
        byte_pos, _, len_posts, len_tf, len_pos = self.postings_dict[term]
        self.index_file.seek(byte_pos)
        postings  = VBEPostings.decode(self.index_file.read(len_posts))
        tf_list   = VBEPostings.decode_tf(self.index_file.read(len_tf))
        positions = _decode_positions(self.index_file.read(len_pos), tf_list)
        return postings, tf_list, positions


# ---------------------------------------------------------------------------
# Phrase query algorithm
# ---------------------------------------------------------------------------

def positional_intersect(postings1, positions1, postings2, positions2):
    """
    Merge two positional postings lists, keeping only documents where
    term2 appears at exactly position+1 relative to term1.

    Parameters
    ----------
    postings1, postings2 : List[int]
        Sorted doc-ID lists for term1 and term2.
    positions1, positions2 : List[List[int]]
        positions1[i] = sorted position list for postings1[i], etc.

    Returns
    -------
    List[Tuple[int, List[int]]]
        (doc_id, phrase_start_positions) for every matching document.
        phrase_start_positions contains each position where term1 starts
        a valid phrase occurrence.

    References
    ----------
    Manning et al., IIR Algorithm 2.4 (positional_intersect).
    """
    results = []
    i, j = 0, 0
    while i < len(postings1) and j < len(postings2):
        d1, d2 = postings1[i], postings2[j]
        if d1 == d2:
            phrase_starts = []
            pp, qq = 0, 0
            pos1 = positions1[i]
            pos2 = positions2[j]
            while pp < len(pos1):
                target = pos1[pp] + 1
                # Advance qq until pos2[qq] >= target
                while qq < len(pos2) and pos2[qq] < target:
                    qq += 1
                if qq < len(pos2) and pos2[qq] == target:
                    phrase_starts.append(pos1[pp])
                pp += 1
            if phrase_starts:
                results.append((d1, phrase_starts))
            i += 1
            j += 1
        elif d1 < d2:
            i += 1
        else:
            j += 1
    return results
