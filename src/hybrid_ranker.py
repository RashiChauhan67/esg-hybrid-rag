"""
ESG RAG System — Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

Combines Dense semantic retrieval and BM25 keyword retrieval results
using Reciprocal Rank Fusion (RRF). RRF is a rank-based fusion method
that is robust to score scale differences between heterogeneous
retrieval systems.

RRF Formula:
    RRF_score(d) = sum_{r in retrievers} 1 / (k + rank_r(d))

where k is a constant (typically 60) that controls the influence of
high-ranked documents, and rank_r(d) is the rank of document d in
retriever r's result list.

Reference:
    Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods". SIGIR 2009.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .dense_retriever import RetrievedPage

logger = logging.getLogger(__name__)


class HybridRanker:
    """Hybrid retrieval via Reciprocal Rank Fusion (RRF).

    Combines ranked result lists from multiple retrievers into a single
    fused ranking. Supports arbitrary numbers of retriever results with
    per-retriever weight multipliers.

    Args:
        rrf_k: The RRF constant k. Larger k reduces the impact of rank
            differences at the top of the list. Default 60 is the
            standard value from the original RRF paper.
        dense_weight: Multiplicative weight for Dense retriever scores.
        bm25_weight: Multiplicative weight for BM25 scores.

    Example:
        >>> ranker = HybridRanker(rrf_k=60, dense_weight=1.0, bm25_weight=0.5)
        >>> results = ranker.fuse(dense_results, bm25_results, top_k=5)
    """

    def __init__(
        self,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        # Legacy alias kept for backward compatibility
        colpali_weight: Optional[float] = None,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError(f"RRF k must be positive, got {rrf_k}")
        self.rrf_k = rrf_k
        # Support old colpali_weight kwarg transparently
        self.dense_weight = colpali_weight if colpali_weight is not None else dense_weight
        self.bm25_weight = bm25_weight

        logger.info(
            "HybridRanker initialized: rrf_k=%d, dense_weight=%.2f, bm25_weight=%.2f",
            rrf_k,
            self.dense_weight,
            bm25_weight,
        )

    def fuse(
        self,
        dense_results: List[RetrievedPage],
        bm25_results: List[RetrievedPage],
        top_k: int = 5,
    ) -> List[RetrievedPage]:
        """Fuse Dense and BM25 results using Reciprocal Rank Fusion.

        Pages appearing in both result lists receive contributions from
        both retrievers. Pages appearing in only one list still receive
        their contribution from that retriever.

        Args:
            dense_results: Ranked results from the Dense retriever.
            bm25_results: Ranked results from BM25 retriever.
            top_k: Number of top results to return after fusion.

        Returns:
            List of RetrievedPage objects ranked by RRF score (descending).
            The ``retriever`` field is set to "hybrid" and ``score`` is
            the RRF score.

        Raises:
            ValueError: If both result lists are empty.
        """
        if not dense_results and not bm25_results:
            raise ValueError("Both retriever result lists are empty.")

        rrf_scores: Dict[str, float] = defaultdict(float)
        page_registry: Dict[str, RetrievedPage] = {}

        # Accumulate RRF scores from Dense results
        for result in dense_results:
            pid = result.page_id
            rrf_score = self.dense_weight / (self.rrf_k + result.rank)
            rrf_scores[pid] += rrf_score
            if pid not in page_registry:
                page_registry[pid] = result

        # Accumulate RRF scores from BM25 results
        for result in bm25_results:
            pid = result.page_id
            rrf_score = self.bm25_weight / (self.rrf_k + result.rank)
            rrf_scores[pid] += rrf_score
            if pid not in page_registry:
                page_registry[pid] = result

        # Sort by RRF score descending
        sorted_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)
        top_pids = sorted_pids[:top_k]

        fused_results: List[RetrievedPage] = []
        for rank, pid in enumerate(top_pids, start=1):
            page = page_registry[pid]
            fused = RetrievedPage(
                page_id=page.page_id,
                company=page.company,
                year=page.year,
                page_number=page.page_number,
                pdf_path=page.pdf_path,
                image=page.image,
                ocr_text=page.ocr_text,
                score=rrf_scores[pid],
                rank=rank,
                retriever="hybrid",
                image_path=page.image_path,
            )
            fused_results.append(fused)

        logger.info(
            "RRF fusion: %d Dense + %d BM25 → %d unique → top %d returned",
            len(dense_results),
            len(bm25_results),
            len(rrf_scores),
            len(fused_results),
        )

        if fused_results:
            logger.debug(
                "Top hybrid result: %s (RRF score=%.6f)",
                fused_results[0].page_id,
                fused_results[0].score,
            )

        return fused_results

    def fuse_multi(
        self,
        results_list: List[Tuple[List[RetrievedPage], float]],
        top_k: int = 5,
    ) -> List[RetrievedPage]:
        """Fuse results from an arbitrary number of retrievers.

        Args:
            results_list: List of (ranked_results, weight) tuples, one per retriever.
            top_k: Number of top results to return.

        Returns:
            Fused ranked list of RetrievedPage objects.

        Raises:
            ValueError: If results_list is empty.
        """
        if not results_list:
            raise ValueError("results_list must not be empty.")

        rrf_scores: Dict[str, float] = defaultdict(float)
        page_registry: Dict[str, RetrievedPage] = {}

        for results, weight in results_list:
            for result in results:
                pid = result.page_id
                rrf_score = weight / (self.rrf_k + result.rank)
                rrf_scores[pid] += rrf_score
                if pid not in page_registry:
                    page_registry[pid] = result

        sorted_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)
        top_pids = sorted_pids[:top_k]

        fused_results: List[RetrievedPage] = []
        for rank, pid in enumerate(top_pids, start=1):
            page = page_registry[pid]
            fused = RetrievedPage(
                page_id=page.page_id,
                company=page.company,
                year=page.year,
                page_number=page.page_number,
                pdf_path=page.pdf_path,
                image=page.image,
                ocr_text=page.ocr_text,
                score=rrf_scores[pid],
                rank=rank,
                retriever="hybrid",
                image_path=page.image_path,
            )
            fused_results.append(fused)

        logger.info(
            "Multi-retriever RRF fusion: %d retrievers → top %d returned",
            len(results_list),
            len(fused_results),
        )
        return fused_results

    def get_config(self) -> dict:
        """Return the current ranker configuration as a dict.

        Returns:
            Dict with rrf_k, dense_weight, bm25_weight.
        """
        return {
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
        }
