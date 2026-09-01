"""
BM25 Sparse (Keyword) Search Index.

Tokenizes complaint narratives and scores them against a query using BM25 —
a rare-word-weighted exact keyword match. Complements dense_index.py's
meaning-based search; see hybrid_search.py for how the two are fused.
"""

import re
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

# Standard English stopwords, hardcoded — no NLTK, no runtime download.
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "of",
    "as", "until", "while", "because",
})

# '§1005.11' etc. must survive tokenization intact. Matched before the
# generic word pattern so the '§' isn't dropped as punctuation and the
# citation isn't split from its number.
_TOKEN_PATTERN = re.compile(r"§\d[\d.]*|[a-z0-9]+")


def tokenize(text: Optional[str]) -> List[str]:
    """
    Lowercases, extracts word/citation tokens, and drops stopwords.

    Regulation citations like '§1005.11' are kept as one token rather than
    having the '§' stripped as punctuation. Dollar amounts and dates need no
    special-casing here — cleaner.py already normalized them to placeholder
    tokens like '[AMOUNT]' and '[DATE]' upstream, in Phase 1.
    """
    if not text:
        return []
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class SparseIndex:
    """
    BM25 keyword index over complaint narratives.

    Rebuilt fresh on every process start rather than persisted to disk:
    tokenizing a few thousand short narratives is sub-second, so persistence
    isn't worth the added complexity (unlike the dense index, where
    embedding is the expensive step worth caching).
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._complaint_ids: List[str] = []

    def build(self, records: List[Dict]) -> None:
        """
        Tokenizes each record's 'cleaned_narrative' and builds the BM25 index.
        Each record must have 'complaint_id' and 'cleaned_narrative' keys.
        """
        self._complaint_ids = [r["complaint_id"] for r in records]
        tokenized_corpus = [tokenize(r["cleaned_narrative"]) for r in records]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Returns up to top_k (complaint_id, bm25_score) pairs for the query,
        highest score first. Complaints with zero score (no shared terms
        with the query) are excluded rather than returned as noise.
        """
        if self._bm25 is None:
            raise RuntimeError("SparseIndex.build() must be called before search().")

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._complaint_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]
