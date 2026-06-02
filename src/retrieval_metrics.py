"""
ESG RAG System — Retrieval Evaluation Metrics

Implements standard information retrieval evaluation metrics:
  - Recall@K: Fraction of relevant pages found in top-K results
  - NDCG@K: Normalized Discounted Cumulative Gain at K
  - MAP: Mean Average Precision
  - MRR: Mean Reciprocal Rank

All metrics accept page_id strings as the atomic relevance unit,
matching the format used in the benchmark.json ground truth file.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalEvalResult:
    """Evaluation results for a single query.

    Attributes:
        query: The query string.
        recall_at_k: Dict mapping K values to Recall@K scores.
        ndcg_at_k: Dict mapping K values to NDCG@K scores.
        average_precision: AP for this query.
        reciprocal_rank: Reciprocal rank of the first relevant result.
        num_relevant: Total number of relevant pages for this query.
        num_retrieved: Total number of retrieved pages.
    """

    query: str
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    average_precision: float = 0.0
    reciprocal_rank: float = 0.0
    num_relevant: int = 0
    num_retrieved: int = 0


# ======================================================================
# Core Metric Functions
# ======================================================================


def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int,
) -> float:
    """Compute Recall@K for a single query.

    Recall@K measures the fraction of relevant documents that appear
    in the top-K retrieved results.

    Recall@K = |{retrieved[:k]} ∩ {relevant}| / |{relevant}|

    Args:
        retrieved: Ordered list of retrieved page_ids (rank 1 first).
        relevant: List of ground-truth relevant page_ids.
        k: Cutoff rank.

    Returns:
        Recall@K score in [0.0, 1.0]. Returns 0.0 if relevant is empty.

    Raises:
        ValueError: If k is not positive.

    Example:
        >>> recall_at_k(["p1", "p2", "p3"], ["p1", "p4"], k=3)
        0.5
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        logger.warning("recall_at_k called with empty relevant set — returning 0.0")
        return 0.0

    relevant_set = set(relevant)
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant_set)
    score = hits / len(relevant_set)
    return score


def ndcg_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int,
) -> float:
    """Compute NDCG@K (Normalized Discounted Cumulative Gain) for a single query.

    NDCG@K measures ranking quality by giving higher credit to relevant
    documents appearing earlier in the ranking. Uses binary relevance
    (1 if page is relevant, 0 otherwise).

    DCG@K  = sum_{i=1}^{K} rel_i / log2(i + 1)
    IDCG@K = DCG of perfect ranking (relevant docs first)
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: Ordered list of retrieved page_ids (rank 1 first).
        relevant: List of ground-truth relevant page_ids.
        k: Cutoff rank.

    Returns:
        NDCG@K score in [0.0, 1.0]. Returns 0.0 if relevant is empty.

    Raises:
        ValueError: If k is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        logger.warning("ndcg_at_k called with empty relevant set — returning 0.0")
        return 0.0

    relevant_set = set(relevant)

    # Compute DCG@K
    dcg = 0.0
    for i, page_id in enumerate(retrieved[:k], start=1):
        if page_id in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG@K: all relevant docs ranked first
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(
    retrieved: List[str],
    relevant: List[str],
) -> float:
    """Compute Average Precision (AP) for a single query.

    AP = mean of precision values at each rank where a relevant
    document is retrieved.

    AP = (1/|R|) * sum_{k: retrieved[k] is relevant} Precision@k

    Args:
        retrieved: Ordered list of retrieved page_ids.
        relevant: List of ground-truth relevant page_ids.

    Returns:
        AP score in [0.0, 1.0]. Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    hits = 0
    precision_sum = 0.0

    for i, page_id in enumerate(retrieved, start=1):
        if page_id in relevant_set:
            hits += 1
            precision_sum += hits / i

    if hits == 0:
        return 0.0

    return precision_sum / len(relevant_set)


def reciprocal_rank(
    retrieved: List[str],
    relevant: List[str],
) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR = 1 / rank of the first relevant document. 0 if none found.

    Args:
        retrieved: Ordered list of retrieved page_ids.
        relevant: List of ground-truth relevant page_ids.

    Returns:
        RR score in [0.0, 1.0].
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    for i, page_id in enumerate(retrieved, start=1):
        if page_id in relevant_set:
            return 1.0 / i
    return 0.0


def mean_average_precision(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]],
) -> float:
    """Compute Mean Average Precision (MAP) over all queries.

    Args:
        all_retrieved: List of retrieved page_id lists, one per query.
        all_relevant: List of relevant page_id lists, one per query.

    Returns:
        MAP score in [0.0, 1.0].

    Raises:
        ValueError: If input lists have different lengths.
    """
    if len(all_retrieved) != len(all_relevant):
        raise ValueError(
            f"Mismatched lengths: {len(all_retrieved)} retrieved vs "
            f"{len(all_relevant)} relevant"
        )
    if not all_retrieved:
        return 0.0

    ap_scores = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    map_score = float(np.mean(ap_scores))
    logger.debug("MAP computed over %d queries: %.4f", len(ap_scores), map_score)
    return map_score


# ======================================================================
# Batch Evaluation
# ======================================================================


def evaluate_retrieval(
    results_list: List[List[str]],
    ground_truth: List[Dict],
    k_values: Optional[List[int]] = None,
) -> Dict:
    """Evaluate retrieval quality across all queries.

    Args:
        results_list: List of retrieved page_id lists, one per query.
            Each inner list should be ordered by rank (best first).
        ground_truth: List of ground truth dicts, one per query.
            Each dict must have keys: "query", "relevant_pages" (list of page_ids).
        k_values: List of K cutoff values for Recall@K and NDCG@K.
            Defaults to [1, 3, 5, 10].

    Returns:
        Dict with per-metric aggregate scores:
            {
                "recall@1": float, "recall@3": float, ...,
                "ndcg@1": float, ...,
                "map": float,
                "mrr": float,
                "num_queries": int,
                "per_query": List[RetrievalEvalResult]
            }

    Raises:
        ValueError: If results_list and ground_truth have different lengths.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if len(results_list) != len(ground_truth):
        raise ValueError(
            f"Mismatch: {len(results_list)} result lists vs "
            f"{len(ground_truth)} ground truth entries"
        )

    per_query_results: List[RetrievalEvalResult] = []
    all_retrieved = []
    all_relevant = []

    for retrieved, gt in zip(results_list, ground_truth):
        query = gt.get("query", "")
        relevant = gt.get("relevant_pages", [])

        all_retrieved.append(retrieved)
        all_relevant.append(relevant)

        result = RetrievalEvalResult(
            query=query,
            num_relevant=len(relevant),
            num_retrieved=len(retrieved),
            average_precision=average_precision(retrieved, relevant),
            reciprocal_rank=reciprocal_rank(retrieved, relevant),
        )

        for k in k_values:
            result.recall_at_k[k] = recall_at_k(retrieved, relevant, k)
            result.ndcg_at_k[k] = ndcg_at_k(retrieved, relevant, k)

        per_query_results.append(result)

    # Aggregate
    aggregated: Dict = {
        "num_queries": len(per_query_results),
        "map": mean_average_precision(all_retrieved, all_relevant),
        "mrr": float(np.mean([r.reciprocal_rank for r in per_query_results])),
        "per_query": per_query_results,
    }

    for k in k_values:
        aggregated[f"recall@{k}"] = float(
            np.mean([r.recall_at_k[k] for r in per_query_results])
        )
        aggregated[f"ndcg@{k}"] = float(
            np.mean([r.ndcg_at_k[k] for r in per_query_results])
        )

    logger.info(
        "Retrieval evaluation complete: %d queries, MAP=%.4f, MRR=%.4f, "
        "Recall@5=%.4f, NDCG@5=%.4f",
        aggregated["num_queries"],
        aggregated["map"],
        aggregated["mrr"],
        aggregated.get("recall@5", 0.0),
        aggregated.get("ndcg@5", 0.0),
    )

    return aggregated
