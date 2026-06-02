"""
ESG RAG System — Evaluation Package

This package provides retrieval metrics (Recall@K, NDCG@K, MAP)
and generation metrics (BERTScore, citation accuracy, faithfulness).
"""

from .retrieval_metrics import (
    recall_at_k,
    ndcg_at_k,
    mean_average_precision,
    evaluate_retrieval,
)
from .generation_metrics import (
    bert_score_evaluate,
    citation_accuracy,
    faithfulness_score,
    evaluate_generation,
)

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "mean_average_precision",
    "evaluate_retrieval",
    "bert_score_evaluate",
    "citation_accuracy",
    "faithfulness_score",
    "evaluate_generation",
]
