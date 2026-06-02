"""
ESG RAG System — Confidence Scoring and Citation Grounding Module

Post-hoc verification and adjustment of LLM-generated answers.
Checks that:
  1. All cited pages were actually in the retrieved context
  2. The cited pages' OCR text contains keywords from the answer
  3. The confidence score is calibrated based on grounding evidence

This module does NOT call an external LLM — it performs lightweight
lexical grounding checks to validate citation quality.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of citation grounding verification.

    Attributes:
        original_confidence: Confidence as reported by the LLM.
        adjusted_confidence: Confidence after grounding adjustment.
        valid_citations: Citations that reference pages actually in context.
        invalid_citations: Citations referencing pages not in context.
        grounding_scores: Per-citation keyword overlap scores (0–1).
        is_grounded: True if at least one valid, grounded citation exists.
        adjustment_reason: Human-readable explanation of any confidence change.
    """

    original_confidence: float
    adjusted_confidence: float
    valid_citations: List
    invalid_citations: List
    grounding_scores: Dict[str, float]
    is_grounded: bool
    adjustment_reason: str


class ConfidenceScorer:
    """Post-hoc citation grounding and confidence calibration.

    Verifies that LLM-generated citations are backed by actual content
    in the retrieved pages and adjusts the confidence score accordingly.

    The grounding score for each citation is based on keyword overlap
    between the answer text and the OCR text of the cited page.

    Args:
        min_grounding_score: Minimum keyword overlap (0–1) for a citation
            to be considered "grounded". Default 0.1 (10% keyword match).
        penalty_per_invalid: Confidence penalty applied for each invalid
            citation (page not in context). Default 0.1.
        reward_per_grounded: Confidence boost for each well-grounded
            citation. Default 0.05.

    Example:
        >>> scorer = ConfidenceScorer()
        >>> result = scorer.score(generated_answer, retrieved_pages)
        >>> print(f"Adjusted confidence: {result.adjusted_confidence:.2f}")
    """

    def __init__(
        self,
        min_grounding_score: float = 0.1,
        penalty_per_invalid: float = 0.1,
        reward_per_grounded: float = 0.05,
    ) -> None:
        self.min_grounding_score = min_grounding_score
        self.penalty_per_invalid = penalty_per_invalid
        self.reward_per_grounded = reward_per_grounded
        logger.info(
            "ConfidenceScorer initialized: min_grounding=%.2f, "
            "penalty=%.2f, reward=%.2f",
            min_grounding_score,
            penalty_per_invalid,
            reward_per_grounded,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        generated_answer,
        retrieved_pages: List,
    ) -> GroundingResult:
        """Verify citations and compute adjusted confidence.

        Args:
            generated_answer: GeneratedAnswer object from LLMGenerator.
            retrieved_pages: List of RetrievedPage objects that were
                provided as context to the LLM.

        Returns:
            GroundingResult with adjusted confidence and grounding details.
        """
        context_map: Dict[str, object] = {
            p.page_id: p for p in retrieved_pages
        }
        valid_ids: Set[str] = set(context_map.keys())

        answer_keywords = self._extract_keywords(generated_answer.answer)

        valid_citations = []
        invalid_citations = []
        grounding_scores: Dict[str, float] = {}
        grounded_count = 0

        for citation in generated_answer.citations:
            if citation.page_id in valid_ids:
                valid_citations.append(citation)
                # Compute keyword overlap with cited page OCR
                page = context_map[citation.page_id]
                gs = self._keyword_overlap(
                    answer_keywords, self._extract_keywords(page.ocr_text)
                )
                grounding_scores[citation.page_id] = gs
                if gs >= self.min_grounding_score:
                    grounded_count += 1
                    logger.debug(
                        "Citation %s: grounded (overlap=%.2f)", citation.page_id, gs
                    )
                else:
                    logger.debug(
                        "Citation %s: low grounding (overlap=%.2f)",
                        citation.page_id,
                        gs,
                    )
            else:
                invalid_citations.append(citation)
                grounding_scores[citation.page_id] = 0.0
                logger.warning(
                    "Citation %r not in retrieved context", citation.page_id
                )

        # Adjust confidence
        original = generated_answer.confidence
        adjusted = original
        reason_parts: List[str] = []

        # Penalty for invalid (hallucinated) citations
        if invalid_citations:
            penalty = len(invalid_citations) * self.penalty_per_invalid
            adjusted = max(0.0, adjusted - penalty)
            reason_parts.append(
                f"-{penalty:.2f} penalty ({len(invalid_citations)} invalid citations)"
            )

        # Reward for well-grounded citations
        if grounded_count > 0:
            reward = min(grounded_count * self.reward_per_grounded, 0.15)
            adjusted = min(1.0, adjusted + reward)
            reason_parts.append(
                f"+{reward:.2f} reward ({grounded_count} grounded citations)"
            )

        # If no citations at all, apply moderate penalty
        if not generated_answer.citations:
            adjusted = max(0.0, adjusted - 0.15)
            reason_parts.append("-0.15 (no citations provided)")

        is_grounded = grounded_count > 0 or (
            valid_citations and not invalid_citations
        )

        reason = "; ".join(reason_parts) if reason_parts else "no adjustment"
        logger.info(
            "Confidence: %.2f → %.2f (%s)", original, adjusted, reason
        )

        return GroundingResult(
            original_confidence=original,
            adjusted_confidence=round(adjusted, 4),
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            grounding_scores=grounding_scores,
            is_grounded=is_grounded,
            adjustment_reason=reason,
        )

    def apply_to_answer(
        self,
        generated_answer,
        retrieved_pages: List,
    ):
        """Apply grounding adjustment and return updated GeneratedAnswer.

        Modifies the answer's confidence in-place and removes invalid
        citations (pages not in context).

        Args:
            generated_answer: GeneratedAnswer to adjust.
            retrieved_pages: Retrieved pages that formed the context.

        Returns:
            The modified GeneratedAnswer with updated confidence.
        """
        result = self.score(generated_answer, retrieved_pages)
        generated_answer.confidence = result.adjusted_confidence
        generated_answer.citations = result.valid_citations
        logger.info(
            "Answer updated: confidence=%.2f, citations=%d",
            generated_answer.confidence,
            len(generated_answer.citations),
        )
        return generated_answer, result

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        """Extract meaningful keywords from a text string.

        Lowercases, removes punctuation, and filters very short tokens.

        Args:
            text: Input text.

        Returns:
            Set of keyword strings.
        """
        if not text:
            return set()
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        # Keep tokens ≥ 3 chars and not pure digits
        return {t for t in tokens if len(t) >= 3 and not t.isdigit()}

    @staticmethod
    def _keyword_overlap(keywords_a: Set[str], keywords_b: Set[str]) -> float:
        """Compute Jaccard-like keyword overlap between two sets.

        Returns the fraction of answer keywords found in the page OCR text.
        Uses intersection over answer keyword count (recall-oriented).

        Args:
            keywords_a: Answer keyword set.
            keywords_b: Page OCR keyword set.

        Returns:
            Float in [0, 1] — higher means more grounded.
        """
        if not keywords_a or not keywords_b:
            return 0.0
        overlap = len(keywords_a & keywords_b)
        return overlap / len(keywords_a)
