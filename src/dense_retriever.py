"""
ESG RAG System — Dense Semantic Retrieval Module

Implements dense semantic retrieval using sentence-transformers
(all-MiniLM-L6-v2 by default). Encodes OCR-extracted page text into
fixed-size embedding vectors and retrieves the most relevant pages
for a text query using cosine similarity.

Why sentence-transformers instead of ColPali:
    - Works reliably on Kaggle (free tier) without kernel restarts
    - Model size ~90 MB vs 5.8 GB for ColPali
    - No GPU required — fast on CPU
    - Well-established retrieval baseline for comparison

Reference:
    Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings
    using Siamese BERT-Networks". EMNLP 2019.
    https://arxiv.org/abs/1908.10084
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPage:
    """A single retrieved page with its retrieval score and metadata.

    Attributes:
        page_id: Unique identifier for the page.
        company: Company name.
        year: Publication year.
        page_number: 1-indexed page number.
        pdf_path: Source PDF path.
        image: PIL Image of the page.
        ocr_text: Extracted OCR text.
        score: Retrieval similarity score (higher = more relevant).
        rank: 1-indexed rank in the result list.
        retriever: Name of the retriever that produced this result.
        image_path: Optional path to saved image on disk.
    """

    page_id: str
    company: str
    year: int
    page_number: int
    pdf_path: str
    image: Image.Image
    ocr_text: str
    score: float
    rank: int
    retriever: str
    image_path: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"RetrievedPage(page_id={self.page_id!r}, rank={self.rank}, "
            f"score={self.score:.4f}, retriever={self.retriever!r})"
        )


class DenseRetriever:
    """Dense semantic retrieval using sentence-transformers.

    Encodes all corpus page texts into dense embedding vectors using a
    sentence-transformer model. At query time, encodes the query into
    the same space and computes cosine similarity against all corpus
    embeddings to retrieve the top-K most relevant pages.

    Corpus embeddings are cached to disk so that re-runs skip the
    expensive encoding step.

    Args:
        model_name: sentence-transformers model identifier.
            Default ``all-MiniLM-L6-v2`` balances speed and quality.
        batch_size: Number of texts to encode per batch.
        embeddings_dir: Directory for saving/loading cached embeddings.

    Example:
        >>> retriever = DenseRetriever()
        >>> retriever.encode_corpus(pages)
        >>> results = retriever.retrieve("What is the CO2 emission target?", top_k=5)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        embeddings_dir: Optional[str | Path] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None

        self._model = None
        self._corpus_pages: List = []
        self._corpus_embeddings: Optional[np.ndarray] = None  # shape (N, D)

        logger.info(
            "DenseRetriever init: model=%s, batch_size=%d",
            model_name,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Model Loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the sentence-transformer model."""
        if self._model is not None:
            return
        logger.info("Loading sentence-transformer model: %s", self.model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.model_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Corpus Encoding
    # ------------------------------------------------------------------

    def encode_corpus(
        self,
        pages: List,
        save_embeddings: bool = True,
    ) -> None:
        """Encode all corpus pages into dense text embeddings.

        Pages with empty OCR text receive a zero vector and will score
        low in retrieval (similar to BM25 behaviour on empty docs).

        Args:
            pages: List of PageImage objects (must have .ocr_text, .page_id).
            save_embeddings: Whether to cache embeddings to disk.
        """
        if not pages:
            raise ValueError("Cannot encode empty corpus.")

        self._corpus_pages = pages

        # Try loading cached embeddings first
        cache_path = self._get_cache_path()
        if cache_path and cache_path.exists():
            logger.info("Loading cached dense embeddings from %s", cache_path)
            cached = self._load_embeddings(cache_path)
            if cached is not None and cached.shape[0] == len(pages):
                self._corpus_embeddings = cached
                logger.info("Loaded %d cached embeddings.", len(pages))
                return
            else:
                logger.warning("Cache size mismatch — re-encoding.")

        self._load_model()

        texts = [p.ocr_text if p.ocr_text and p.ocr_text.strip() else " " for p in pages]

        logger.info("Encoding %d pages with batch_size=%d...", len(texts), self.batch_size)
        self._corpus_embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # enables fast dot-product cosine sim
            convert_to_numpy=True,
        )
        logger.info(
            "Encoded corpus: shape=%s", self._corpus_embeddings.shape
        )

        if save_embeddings and cache_path:
            self._save_embeddings(self._corpus_embeddings, cache_path)
            logger.info("Cached embeddings to %s", cache_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPage]:
        """Retrieve the top-K most relevant pages for a text query.

        Args:
            query: Natural-language query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievedPage ranked by cosine similarity (descending).

        Raises:
            RuntimeError: If encode_corpus() has not been called yet.
        """
        if self._corpus_embeddings is None or not self._corpus_pages:
            raise RuntimeError(
                "Corpus is not encoded. Call encode_corpus() first."
            )

        self._load_model()

        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )  # shape (1, D)

        # Cosine similarity via dot product (embeddings already normalised)
        scores = (self._corpus_embeddings @ query_embedding.T).squeeze()  # (N,)

        top_k_actual = min(top_k, len(self._corpus_pages))
        top_indices = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[RetrievedPage] = []
        for rank, idx in enumerate(top_indices, start=1):
            page = self._corpus_pages[idx]
            results.append(
                RetrievedPage(
                    page_id=page.page_id,
                    company=getattr(page, "company", "unknown"),
                    year=getattr(page, "year", 0),
                    page_number=getattr(page, "page_number", 0),
                    pdf_path=str(getattr(page, "pdf_path", "")),
                    image=getattr(page, "image", None),
                    ocr_text=getattr(page, "ocr_text", ""),
                    score=float(scores[idx]),
                    rank=rank,
                    retriever="dense",
                    image_path=getattr(page, "image_path", None),
                )
            )

        logger.debug(
            "Dense retrieval: top result=%s (score=%.4f)",
            results[0].page_id if results else "none",
            results[0].score if results else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # Cache Utilities
    # ------------------------------------------------------------------

    def _get_cache_path(self) -> Optional[Path]:
        if not self.embeddings_dir:
            return None
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        model_slug = self.model_name.replace("/", "_").replace("-", "_")
        return self.embeddings_dir / f"dense_{model_slug}_embeddings.pkl"

    def _save_embeddings(self, embeddings: np.ndarray, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(embeddings, f)

    def _load_embeddings(self, path: Path) -> Optional[np.ndarray]:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load cached embeddings: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_index_stats(self) -> dict:
        """Return statistics about the indexed corpus."""
        return {
            "model_name": self.model_name,
            "num_pages": len(self._corpus_pages),
            "embedding_dim": (
                self._corpus_embeddings.shape[1]
                if self._corpus_embeddings is not None
                else None
            ),
            "cache_path": str(self._get_cache_path()),
        }
