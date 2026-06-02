"""
ESG RAG System — BM25 Keyword Retrieval Module

Implements BM25 (Best Matching 25) keyword-based retrieval over
OCR-extracted text from ESG PDF pages. Used as a complementary
retriever alongside ColPali visual retrieval in the hybrid pipeline.

BM25 is particularly effective for queries that contain specific
keywords, numbers, or named entities (e.g. "Scope 3 emissions 2023").

Uses the rank_bm25 library with NLTK-based tokenization and
optional stopword removal.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from rank_bm25 import BM25Okapi

from .dense_retriever import RetrievedPage

logger = logging.getLogger(__name__)

# Bundled fallback English stopwords — used when NLTK download fails (e.g., SSL errors)
_FALLBACK_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "each", "all", "any", "this", "that",
    "these", "those", "its", "it", "as", "if", "then", "than", "when",
    "while", "where", "which", "who", "whom", "how", "what", "why",
    "we", "our", "us", "you", "your", "he", "she", "they", "their",
    "about", "above", "after", "before", "between", "during", "into",
    "through", "under", "until", "up", "over", "such", "also", "more",
    "other", "own", "same", "just", "because", "per", "however",
}


def _load_nltk_stopwords(language: str) -> Set[str]:
    """Load NLTK stopwords with graceful SSL fallback.

    Attempts to download NLTK stopwords. If the download fails due
    to SSL or network issues, falls back to the bundled English stopword list.

    Args:
        language: Language for NLTK stopwords (e.g., 'english').

    Returns:
        Set of stopword strings.
    """
    try:
        import nltk
        # First try to find locally installed data
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        from nltk.corpus import stopwords
        return set(stopwords.words(language))

    except Exception as exc:
        logger.warning(
            "NLTK stopwords not available (%s). Using built-in fallback stopwords.", exc
        )
        return _FALLBACK_STOPWORDS.copy()


class BM25Retriever:
    """BM25 keyword retrieval over OCR-extracted ESG page text.

    Tokenizes each page's OCR text, builds a BM25Okapi index, and
    retrieves the most relevant pages for a query. Useful as a
    complementary signal to ColPali's visual retrieval, especially
    for keyword-heavy queries.

    Args:
        language: Language for NLTK stopwords ("english" or other NLTK languages).
        remove_stopwords: Whether to remove stopwords before indexing.
        min_token_length: Minimum character length for tokens (filters noise).

    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.index_corpus(pages)
        >>> results = retriever.retrieve("carbon emissions target 2023", top_k=5)
    """

    def __init__(
        self,
        language: str = "english",
        remove_stopwords: bool = True,
        min_token_length: int = 2,
    ) -> None:
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

        if remove_stopwords:
            self._stopwords = _load_nltk_stopwords(language)
        else:
            self._stopwords = set()

        self._bm25: Optional[BM25Okapi] = None
        self._corpus_pages: List = []
        self._tokenized_corpus: List[List[str]] = []

        logger.info(
            "BM25Retriever initialized: language=%s, remove_stopwords=%s, "
            "min_token_length=%d",
            language,
            remove_stopwords,
            min_token_length,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_corpus(self, pages: List) -> None:
        """Build the BM25 index from a list of PageImage objects.

        Tokenizes each page's ocr_text field and builds a BM25Okapi
        index over the full corpus.

        Args:
            pages: List of PageImage objects with ocr_text populated.

        Raises:
            ValueError: If pages list is empty or all OCR texts are empty.
        """
        if not pages:
            raise ValueError("Cannot index empty corpus.")

        self._corpus_pages = pages
        self._tokenized_corpus = []

        empty_count = 0
        for page in pages:
            tokens = self._tokenize(page.ocr_text)
            if not tokens:
                empty_count += 1
                tokens = ["<empty>"]  # BM25 requires non-empty token lists
            self._tokenized_corpus.append(tokens)

        if empty_count > 0:
            logger.warning(
                "%d pages had empty OCR text (will receive low BM25 scores).",
                empty_count,
            )

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(
            "BM25 index built: %d pages, avg tokens/page=%.1f",
            len(pages),
            np.mean([len(t) for t in self._tokenized_corpus]),
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPage]:
        """Retrieve top-K most relevant pages using BM25 scoring.

        Tokenizes the query using the same preprocessing as the corpus,
        then computes BM25 scores and returns the top-K pages.

        Args:
            query: Natural language query string.
            top_k: Number of top pages to return.

        Returns:
            List of RetrievedPage objects sorted by score (descending).

        Raises:
            RuntimeError: If index_corpus() has not been called yet.
        """
        if self._bm25 is None or not self._corpus_pages:
            raise RuntimeError(
                "BM25 index not built. Call index_corpus() before retrieve()."
            )

        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.warning("Query tokenized to empty token list: %r", query)
            query_tokens = query.lower().split()

        logger.debug(
            "BM25 retrieving for query tokens: %s (top_k=%d)", query_tokens, top_k
        )

        scores: np.ndarray = self._bm25.get_scores(query_tokens)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievedPage] = []
        for rank, idx in enumerate(top_indices, start=1):
            page = self._corpus_pages[idx]
            results.append(
                RetrievedPage(
                    page_id=page.page_id,
                    company=page.company,
                    year=page.year,
                    page_number=page.page_number,
                    pdf_path=page.pdf_path,
                    image=page.image,
                    ocr_text=page.ocr_text,
                    score=float(scores[idx]),
                    rank=rank,
                    retriever="bm25",
                    image_path=getattr(page, "image_path", None),
                )
            )

        logger.debug(
            "BM25 top result: %s (score=%.4f)",
            results[0].page_id if results else "none",
            results[0].score if results else 0.0,
        )
        return results

    def get_index_stats(self) -> dict:
        """Return statistics about the current BM25 index.

        Returns:
            Dict with total_pages, avg_tokens, vocab_size.
        """
        if self._bm25 is None:
            return {"indexed": False}

        vocab_size = len(self._bm25.idf) if hasattr(self._bm25, "idf") else 0
        avg_tokens = (
            np.mean([len(t) for t in self._tokenized_corpus])
            if self._tokenized_corpus
            else 0.0
        )
        return {
            "indexed": True,
            "total_pages": len(self._corpus_pages),
            "avg_tokens_per_page": float(avg_tokens),
            "vocab_size": vocab_size,
        }

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize a text string for BM25 indexing.

        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Word tokenize (NLTK)
        4. Filter stopwords and short tokens

        Args:
            text: Raw text string to tokenize.

        Returns:
            List of normalized, filtered tokens.
        """
        if not text or not text.strip():
            return []

        # Lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize — prefer NLTK punkt, fall back to simple split
        try:
            import nltk as _nltk
            tokens = _nltk.word_tokenize(text)
        except Exception:
            tokens = text.split()

        # Filter
        tokens = [
            t for t in tokens
            if len(t) >= self.min_token_length
            and t not in self._stopwords
            and not t.isdigit()  # Remove pure numbers (keep mixed like "co2")
        ]

        return tokens
