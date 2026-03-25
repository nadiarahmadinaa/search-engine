"""
Patricia Trie (Compressed Radix Trie) for TP2 term dictionary.

A Patricia Trie collapses long single-child chains into one node whose
edge carries a multi-character label, giving O(m) lookup (m = key length)
while enabling prefix search and wildcard queries — operations that a
plain Python dict cannot do.

Node structure
--------------
    edge_label  : str   — the compressed edge string (≥ 1 character)
    is_terminal : bool  — True if this node represents a complete key
    value       : Any   — stored payload (postings metadata tuple)
    children    : dict  — {first_char: PatriciaNode}

Dict-compatible API
-------------------
    trie[key]        → value  (raises KeyError if absent)
    trie[key] = v    → insert / update
    key in trie      → bool
    len(trie)        → number of stored keys

Extended API
------------
    get_terms_by_prefix(prefix) → [(term, value), ...]
    wildcard_search(pattern)    → [(term, value), ...]  (fnmatch patterns)

References
----------
Morrison, D.R. (1968) — "PATRICIA — Practical Algorithm To Retrieve
    Information Coded In Alphanumeric", JACM 15(4): 514–534.
Knuth, TAOCP Vol. 3 §6.3; Sedgewick & Wayne, Algorithms 4e Ch. 5.2.
"""

import fnmatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of strings a and b."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class PatriciaNode:
    """A single node in the Patricia Trie."""
    __slots__ = ('edge_label', 'is_terminal', 'value', 'children')

    def __init__(self, edge_label: str = '', is_terminal: bool = False,
                 value=None):
        self.edge_label  = edge_label
        self.is_terminal = is_terminal
        self.value       = value
        self.children    = {}          # first char of child edge → PatriciaNode


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class PatriciaTrie:
    """
    Compressed radix trie with dict-compatible API.

    Keys must be strings (or convertible via str()).  Values can be
    any Python object.  None is not a valid value (used as sentinel
    for "absent").
    """

    def __init__(self):
        self.root   = PatriciaNode()
        self._size  = 0   # number of stored keys

    # ------------------------------------------------------------------
    # Core insert
    # ------------------------------------------------------------------

    def insert(self, key: str, value) -> None:
        """
        Insert key → value into the trie.

        If key already exists the value is updated (size unchanged).
        """
        node      = self.root
        remaining = key

        while remaining:
            fc = remaining[0]
            if fc not in node.children:
                # No child for this first character → create leaf
                node.children[fc] = PatriciaNode(remaining, True, value)
                self._size += 1
                return

            child  = node.children[fc]
            common = _common_prefix_len(remaining, child.edge_label)

            if common == len(child.edge_label):
                # Edge fully matched → descend deeper
                remaining = remaining[common:]
                node      = child
                # If remaining is now empty the while-condition handles it
            else:
                # Partial edge match → split the edge at `common`
                split          = PatriciaNode(child.edge_label[:common])
                child.edge_label = child.edge_label[common:]
                split.children[child.edge_label[0]] = child
                node.children[fc] = split

                if common < len(remaining):
                    # New key diverges after the common prefix
                    new_leaf = PatriciaNode(remaining[common:], True, value)
                    split.children[remaining[common]] = new_leaf
                    self._size += 1
                else:
                    # New key IS the common prefix → split node is terminal
                    split.is_terminal = True
                    split.value       = value
                    self._size       += 1
                return

        # remaining is empty → current node should be the terminal
        if not node.is_terminal:
            self._size += 1
        node.is_terminal = True
        node.value       = value

    # ------------------------------------------------------------------
    # Exact lookup
    # ------------------------------------------------------------------

    def get(self, key: str):
        """Return value for key, or None if absent."""
        node      = self.root
        remaining = key

        while remaining:
            fc = remaining[0]
            if fc not in node.children:
                return None
            child  = node.children[fc]
            common = _common_prefix_len(remaining, child.edge_label)
            if common < len(child.edge_label):
                return None          # prefix doesn't fully match any edge
            remaining = remaining[common:]
            node      = child

        return node.value if node.is_terminal else None

    # ------------------------------------------------------------------
    # Prefix search
    # ------------------------------------------------------------------

    def get_terms_by_prefix(self, prefix: str):
        """
        Return all (term, value) pairs whose key starts with prefix.

        Uses trie structure for O(|prefix| + output) time — no full
        scan required for short prefixes.
        """
        node             = self.root
        accumulated      = ""   # full edge path consumed so far
        remaining        = prefix

        while remaining:
            fc = remaining[0]
            if fc not in node.children:
                return []
            child  = node.children[fc]
            common = _common_prefix_len(remaining, child.edge_label)

            if common == len(remaining):
                # remaining exhausted (may be mid-edge) → node is a prefix match
                accumulated += child.edge_label   # include the FULL edge
                node         = child
                remaining    = ""
            elif common == len(child.edge_label):
                # Edge fully consumed → continue deeper
                accumulated += child.edge_label
                remaining    = remaining[common:]
                node         = child
            else:
                # Mismatch: no key shares this prefix
                return []

        results = []
        self._collect(node, accumulated, results)
        return results

    # ------------------------------------------------------------------
    # Wildcard search
    # ------------------------------------------------------------------

    def wildcard_search(self, pattern: str):
        """
        Return all (term, value) pairs whose key matches the fnmatch
        pattern (supports * and ?).

        Uses the prefix before the first wildcard to narrow candidates
        with the trie before applying the full pattern.
        """
        if '*' not in pattern and '?' not in pattern:
            val = self.get(pattern)
            return [(pattern, val)] if val is not None else []

        # Extract the longest literal prefix (before any wildcard)
        prefix = ''
        for ch in pattern:
            if ch in ('*', '?'):
                break
            prefix += ch

        candidates = self.get_terms_by_prefix(prefix)
        return [(t, v) for t, v in candidates if fnmatch.fnmatch(t, pattern)]

    # ------------------------------------------------------------------
    # Internal collector
    # ------------------------------------------------------------------

    def _collect(self, node: PatriciaNode, prefix: str, results: list) -> None:
        """Depth-first collect of all (term, value) pairs under node."""
        if node.is_terminal:
            results.append((prefix, node.value))
        for child in node.children.values():
            self._collect(child, prefix + child.edge_label, results)

    # ------------------------------------------------------------------
    # Dict-compatible API
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        val = self.get(str(key))
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        self.insert(str(key), value)

    def __contains__(self, key):
        return self.get(str(key)) is not None

    def __len__(self):
        return self._size

    def items(self):
        """Iterate all (term, value) pairs (like dict.items())."""
        results = []
        self._collect(self.root, '', results)
        return results
