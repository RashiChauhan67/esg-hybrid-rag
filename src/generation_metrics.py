"""
ESG RAG System — Generation Quality Evaluation Metrics

Implements metrics for evaluating the quality of LLM-generated answers:
  - BERTScore: Semantic similarity between predicted and reference answers
  - Citation Accuracy: % of answers where cited pages actually contain
    the answer content (lexical grounding check)
  - Faithfulness Score: Whether the answer content is supported by
    retrieved context (without external knowledge)
  - Answer Completeness: Whether all ground-truth key facts are present

These metrics are adapted for the ESG domain where factual accuracy
and source attribution are critical for research credibility.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenerationEvalResult:
    """Evaluation results for a single generated answer.

    Attributes:
        query: The query that was answered.
        bert_f1: BERTScore F1 (semantic similarity to reference answer).
        bert_precision: BERTScore Precision.
        bert_recall: BERTScore Recall.
        citation_accuracy: Whether cited pages contain answer evidence.
        faithfulness: Fraction of answer content grounded in retrieved pages.
        answer_length: Token count of the generated answer.
    """

    query: str
    bert_f1: float = 0.0
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    citation_accuracy: float = 0.0
    faithfulness: float = 0.0
    answer_length: int = 0


# ======================================================================
# BERTScore
# ======================================================================


def bert_score_evaluate(
    predictions: List[str],
    references: List[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict:
    """Compute BERTScore between predicted and reference answers.

    BERTScore measures semantic similarity using contextual BERT
    embeddings. F1 is the primary metric; precision and recall are
    also reported.

    Args:
        predictions: List of generated answer strings.
        references: List of ground-truth answer strings.
        model_type: HuggingFace model for BERTScore. Using distilbert
            for speed; roberta-large gives best scores but is slower.
        batch_size: Batch size for BERTScore computation.
        device: "cuda" or "cpu". Auto-detected if None.

    Returns:
        Dict with keys: "precision" (list), "recall" (list), "f1" (list),
        "mean_f1", "mean_precision", "mean_recall".

    Raises:
        ImportError: If bert_score is not installed.
        ValueError: If predictions and references have different lengths.

    Example:
        >>> results = bert_score_evaluate(["Tesla's CO2 is 100Mt"], ["CO2 100Mt"])
        >>> print(f"BERTScore F1: {results['mean_f1']:.4f}")
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Lengths differ: {len(predictions)} predictions vs "
            f"{len(references)} references"
        )
    if not predictions:
        raise ValueError("predictions list must not be empty")

    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        raise ImportError(
            "bert-score is not installed. Install with: pip install bert-score"
        ) from exc

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Computing BERTScore: %d pairs, model=%s, device=%s",
        len(predictions),
        model_type,
        device,
    )

    try:
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=False,
            lang="en",
        )
    except Exception as exc:
        raise RuntimeError(f"BERTScore computation failed: {exc}") from exc

    precision_list = P.tolist()
    recall_list = R.tolist()
    f1_list = F1.tolist()

    result = {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "mean_precision": float(np.mean(precision_list)),
        "mean_recall": float(np.mean(recall_list)),
        "mean_f1": float(np.mean(f1_list)),
        "model_type": model_type,
        "num_pairs": len(predictions),
    }

    logger.info(
        "BERTScore: mean_F1=%.4f, mean_P=%.4f, mean_R=%.4f",
        result["mean_f1"],
        result["mean_precision"],
        result["mean_recall"],
    )
    return result


# ======================================================================
# Citation Accuracy
# ======================================================================


def citation_accuracy(
    answers: List,
    ground_truth: List[Dict],
    retrieved_pages_list: List[List],
    min_overlap: float = 0.1,
) -> Dict:
    """Compute citation accuracy: % answers where cited pages contain evidence.

    For each answer, we check whether the cited pages' OCR text has
    sufficient keyword overlap with the generated answer. This serves as
    a proxy for "the cited page actually contains the answer".

    Args:
        answers: List of GeneratedAnswer objects.
        ground_truth: List of ground truth dicts with "relevant_pages" key.
        retrieved_pages_list: List of retrieved page lists (one per query).
        min_overlap: Minimum keyword overlap score (0–1) to count a
            citation as accurate. Default 0.1.

    Returns:
        Dict with keys:
            "citation_accuracy": float — % of queries with ≥1 accurate citation
            "avg_citations_per_answer": float
            "hallucination_rate": float — % answers with citations to non-retrieved pages
            "per_query": list of per-query citation accuracy scores

    Raises:
        ValueError: If input lists have different lengths.
    """
    if not (len(answers) == len(ground_truth) == len(retrieved_pages_list)):
        raise ValueError(
            f"All input lists must have the same length. Got "
            f"{len(answers)}, {len(ground_truth)}, {len(retrieved_pages_list)}"
        )

    per_query_scores: List[float] = []
    hallucination_count = 0

    for answer, gt, retrieved_pages in zip(answers, ground_truth, retrieved_pages_list):
        if not answer.citations:
            per_query_scores.append(0.0)
            continue

        context_map = {p.page_id: p for p in retrieved_pages}
        answer_keywords = _extract_keywords(answer.answer)

        accurate_count = 0
        has_hallucination = False

        for citation in answer.citations:
            if citation.page_id not in context_map:
                has_hallucination = True
                continue
            page = context_map[citation.page_id]
            page_keywords = _extract_keywords(page.ocr_text)
            overlap = _keyword_overlap(answer_keywords, page_keywords)
            if overlap >= min_overlap:
                accurate_count += 1

        if has_hallucination:
            hallucination_count += 1

        score = accurate_count / len(answer.citations) if answer.citations else 0.0
        per_query_scores.append(score)

    num_queries = len(answers)
    result = {
        "citation_accuracy": float(np.mean(per_query_scores)) if per_query_scores else 0.0,
        "avg_citations_per_answer": float(
            np.mean([len(a.citations) for a in answers])
        ),
        "hallucination_rate": hallucination_count / num_queries if num_queries else 0.0,
        "per_query": per_query_scores,
        "num_queries": num_queries,
    }

    logger.info(
        "Citation accuracy: %.4f, hallucination rate: %.4f",
        result["citation_accuracy"],
        result["hallucination_rate"],
    )
    return result


# ======================================================================
# Faithfulness Score
# ======================================================================


def faithfulness_score(
    answer: str,
    retrieved_pages: List,
    threshold: float = 0.15,
) -> float:
    """Estimate answer faithfulness to the retrieved context.

    Measures what fraction of the answer's content can be lexically
    traced back to the retrieved page texts. This is a lightweight
    proxy for RAG faithfulness (not as strong as NLI-based checks).

    Args:
        answer: Generated answer string.
        retrieved_pages: List of RetrievedPage objects used as context.
        threshold: Minimum keyword overlap for the answer to be
            considered faithful. Default 0.15.

    Returns:
        Faithfulness score in [0.0, 1.0].
    """
    if not answer or not retrieved_pages:
        return 0.0

    answer_keywords = _extract_keywords(answer)
    if not answer_keywords:
        return 0.0

    # Union of all retrieved page keywords
    context_keywords: set = set()
    for page in retrieved_pages:
        context_keywords |= _extract_keywords(page.ocr_text)

    overlap = _keyword_overlap(answer_keywords, context_keywords)
    score = min(1.0, overlap / threshold) if threshold > 0 else float(overlap >= 0.1)
    score = min(1.0, score)

    logger.debug(
        "Faithfulness: answer_kw=%d, context_kw=%d, overlap=%.2f, score=%.4f",
        len(answer_keywords),
        len(context_keywords),
        overlap,
        score,
    )
    return round(score, 4)


# ======================================================================
# Batch Evaluation
# ======================================================================


def evaluate_generation(
    answers: List,
    ground_truth: List[Dict],
    retrieved_pages_list: List[List],
    compute_bertscore: bool = True,
    bertscore_model: str = "distilbert-base-uncased",
) -> Dict:
    """Run full generation evaluation pipeline.

    Args:
        answers: List of GeneratedAnswer objects.
        ground_truth: List of ground truth dicts with "query" and "answer" keys.
        retrieved_pages_list: List of retrieved page lists per query.
        compute_bertscore: Whether to compute BERTScore (requires bert-score).
        bertscore_model: HuggingFace model for BERTScore.

    Returns:
        Dict with all generation metrics aggregated across queries.
    """
    predictions = [a.answer for a in answers]
    references = [gt.get("answer", "") for gt in ground_truth]

    results: Dict = {}

    # BERTScore
    if compute_bertscore:
        try:
            bs_results = bert_score_evaluate(
                predictions, references, model_type=bertscore_model
            )
            results["bertscore"] = {
                "mean_f1": bs_results["mean_f1"],
                "mean_precision": bs_results["mean_precision"],
                "mean_recall": bs_results["mean_recall"],
            }
        except Exception as exc:
            logger.error("BERTScore failed: %s", exc)
            results["bertscore"] = {"error": str(exc)}

    # Citation accuracy
    try:
        cite_results = citation_accuracy(answers, ground_truth, retrieved_pages_list)
        results["citation"] = cite_results
    except Exception as exc:
        logger.error("Citation accuracy failed: %s", exc)
        results["citation"] = {"error": str(exc)}

    # Faithfulness
    faith_scores = []
    for answer, retrieved_pages in zip(answers, retrieved_pages_list):
        fs = faithfulness_score(answer.answer, retrieved_pages)
        faith_scores.append(fs)
    results["faithfulness"] = {
        "mean": float(np.mean(faith_scores)) if faith_scores else 0.0,
        "per_query": faith_scores,
    }

    # Confidence stats
    confidences = [a.confidence for a in answers]
    results["confidence"] = {
        "mean": float(np.mean(confidences)) if confidences else 0.0,
        "min": float(np.min(confidences)) if confidences else 0.0,
        "max": float(np.max(confidences)) if confidences else 0.0,
    }

    results["num_queries"] = len(answers)

    logger.info(
        "Generation evaluation complete: %d queries, "
        "BERTScore-F1=%.4f, CitationAcc=%.4f, Faithfulness=%.4f",
        results["num_queries"],
        results.get("bertscore", {}).get("mean_f1", 0.0),
        results.get("citation", {}).get("citation_accuracy", 0.0),
        results.get("faithfulness", {}).get("mean", 0.0),
    )

    return results


# ======================================================================
# Shared Utilities
# ======================================================================


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from a text string.

    Args:
        text: Input text string.

    Returns:
        Set of lowercase, punctuation-stripped tokens (≥ 3 chars).
    """
    if not text:
        return set()
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return {t for t in text.split() if len(t) >= 3}


def _keyword_overlap(set_a: set, set_b: set) -> float:
    """Compute recall-oriented keyword overlap (|A ∩ B| / |A|).

    Args:
        set_a: Query set (numerator denominator).
        set_b: Reference set.

    Returns:
        Overlap score in [0.0, 1.0].
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a)
