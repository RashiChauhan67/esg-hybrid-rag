"""
ESG RAG System — Multimodal LLM Generation Module

Passes retrieved ESG page images and the user query to a multimodal
LLM (Google Gemini 2.5 Flash or OpenAI GPT-4o) to generate structured
answers with page citations and a confidence score.

The generation prompt instructs the LLM to:
    1. Answer the query based on the provided page images
    2. Cite the specific pages (page_id, company, year, page_number)
    3. Provide a confidence score from 0.0 to 1.0

Output is a structured JSON: {"answer": str, "citations": list, "confidence": float}
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a cited source page in the generated answer.

    Attributes:
        page_id: Unique identifier of the cited page.
        company: Company name.
        year: Publication year.
        page_number: 1-indexed page number.
        relevance_note: Optional LLM-generated note on why this page was cited.
    """

    page_id: str
    company: str
    year: int
    page_number: int
    relevance_note: str = ""


@dataclass
class GeneratedAnswer:
    """Structured output from the multimodal LLM generation step.

    Attributes:
        answer: The natural language answer to the user query.
        citations: List of Citation objects identifying source pages.
        confidence: Confidence score in [0.0, 1.0].
        query: The original query that prompted this answer.
        llm_backend: Which LLM was used ("gemini" or "openai").
        raw_response: The raw LLM response string (for debugging).
    """

    answer: str
    citations: List[Citation]
    confidence: float
    query: str
    llm_backend: str
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the answer to a JSON-compatible dictionary.

        Returns:
            Dict with answer, citations (as dicts), confidence, query, llm_backend.
        """
        return {
            "answer": self.answer,
            "citations": [
                {
                    "page_id": c.page_id,
                    "company": c.company,
                    "year": c.year,
                    "page_number": c.page_number,
                    "relevance_note": c.relevance_note,
                }
                for c in self.citations
            ],
            "confidence": round(self.confidence, 4),
            "query": self.query,
            "llm_backend": self.llm_backend,
        }

    def __repr__(self) -> str:
        return (
            f"GeneratedAnswer(confidence={self.confidence:.2f}, "
            f"citations={len(self.citations)}, "
            f"answer={self.answer[:80]!r}...)"
        )


# ======================================================================
# Prompt Templates
# ======================================================================

SYSTEM_PROMPT = """You are an expert ESG (Environmental, Social, and Governance) 
analyst. You have been provided with page images from sustainability reports.
Your task is to answer the user's question based ONLY on the visual information 
in the provided pages.

RESPONSE FORMAT — You MUST respond with a JSON object only, no other text:
{
    "answer": "Your detailed answer here, referencing specific data from the pages",
    "citations": [
        {
            "page_id": "company_year_pN",
            "company": "Company Name",
            "year": 2023,
            "page_number": N,
            "relevance_note": "This page contains the emissions table referenced"
        }
    ],
    "confidence": 0.85
}

RULES:
1. Answer ONLY from the provided page images — do not use external knowledge
2. Cite every page that contributed to your answer
3. If the answer is not found in any page, set confidence < 0.3 and explain
4. Confidence scale: 0.9-1.0 = very certain, 0.7-0.9 = fairly certain, 
   0.5-0.7 = somewhat uncertain, <0.5 = uncertain or not found
"""

USER_PROMPT_TEMPLATE = """Question: {query}

I have provided {num_pages} page image(s) from ESG sustainability reports as context.
The pages are from: {page_summary}

Please analyze the pages and answer the question in the required JSON format."""


class LLMGenerator:
    """Multimodal LLM-based answer generation for ESG RAG.

    Supports two backends:
    - "gemini": Google Gemini 2.5 Flash (recommended — native image support)
    - "openai": OpenAI GPT-4o (via base64 image encoding)

    Args:
        backend: LLM backend to use ("gemini" or "openai").
        model_name: Specific model name. Defaults to the recommended model
            for each backend.
        max_context_pages: Maximum number of page images to include in a
            single LLM call. Reduces token/cost for large retrievals.
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative).
            Keep low for factual ESG data extraction.

    Example:
        >>> generator = LLMGenerator(backend="gemini")
        >>> answer = generator.generate("What is Tesla's Scope 1 emissions?", pages)
        >>> print(answer.to_dict())
    """

    def __init__(
        self,
        backend: str = "gemini",
        model_name: Optional[str] = None,
        max_context_pages: int = 5,
        temperature: float = 0.1,
    ) -> None:
        backend = backend.lower()
        if backend not in ("gemini", "openai"):
            raise ValueError(f"Unsupported backend: {backend!r}. Use 'gemini' or 'openai'.")

        self.backend = backend
        self.temperature = temperature
        self.max_context_pages = max_context_pages

        # Set default model names
        if model_name:
            self.model_name = model_name
        elif backend == "gemini":
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

        self._client = None
        logger.info(
            "LLMGenerator initialized: backend=%s, model=%s, temperature=%.2f",
            backend,
            self.model_name,
            temperature,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_pages: List,
        system_prompt: Optional[str] = None,
    ) -> GeneratedAnswer:
        """Generate a structured answer from retrieved ESG page images.

        Sends the query and page images to the configured LLM and parses
        the structured JSON response.

        Args:
            query: Natural language question to answer.
            retrieved_pages: List of RetrievedPage objects (ordered by rank).
            system_prompt: Optional custom system prompt. Defaults to the
                built-in ESG analyst prompt.

        Returns:
            GeneratedAnswer with answer, citations, and confidence score.

        Raises:
            RuntimeError: If the LLM call fails or returns unparseable output.
            ValueError: If retrieved_pages is empty.
        """
        if not retrieved_pages:
            raise ValueError("retrieved_pages must not be empty.")

        # Limit to max_context_pages
        context_pages = retrieved_pages[: self.max_context_pages]
        logger.info(
            "Generating answer for query=%r using %d pages via %s/%s",
            query[:60],
            len(context_pages),
            self.backend,
            self.model_name,
        )

        sys_prompt = system_prompt or SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(query, context_pages)

        if self.backend == "gemini":
            raw_response = self._call_gemini(sys_prompt, user_prompt, context_pages)
        else:
            raw_response = self._call_openai(sys_prompt, user_prompt, context_pages)

        answer = self._parse_response(raw_response, query, context_pages)
        logger.info(
            "Answer generated: confidence=%.2f, citations=%d",
            answer.confidence,
            len(answer.citations),
        )
        return answer

    # ------------------------------------------------------------------
    # Backend Implementations
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        pages: List,
    ) -> str:
        """Call Google Gemini 2.5 Flash with text + image content.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: User query and context description.
            pages: List of RetrievedPage objects to include as images.

        Returns:
            Raw string response from the model.

        Raises:
            RuntimeError: If the API call fails.
        """
        self._init_gemini()

        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )

        # Build content: text prompt + PIL images
        content = [user_prompt]
        for i, page in enumerate(pages):
            content.append(f"\n--- Page {i+1}: {page.page_id} ---\n")
            content.append(page.image)  # Gemini accepts PIL.Image directly

        try:
            response = model.generate_content(content)
            raw = response.text
            logger.debug("Gemini raw response (first 300 chars): %s", raw[:300])
            return raw
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        pages: List,
    ) -> str:
        """Call OpenAI GPT-4o with text + base64-encoded images.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: User query and context description.
            pages: List of RetrievedPage objects to include as images.

        Returns:
            Raw string response from the model.

        Raises:
            RuntimeError: If the API call fails.
        """
        self._init_openai()

        # Build message content with interleaved text and images
        content: List[Dict] = [{"type": "text", "text": user_prompt}]

        for i, page in enumerate(pages):
            b64_image = self._pil_to_base64(page.image)
            content.append(
                {
                    "type": "text",
                    "text": f"\n--- Page {i+1}: {page.page_id} ---",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                        "detail": "high",
                    },
                }
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            logger.debug("OpenAI raw response (first 300 chars): %s", raw[:300])
            return raw
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Response Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw_response: str,
        query: str,
        context_pages: List,
    ) -> GeneratedAnswer:
        """Parse the LLM JSON response into a GeneratedAnswer object.

        Tries strict JSON parsing first, then falls back to regex extraction
        for robustness against minor LLM formatting errors.

        Args:
            raw_response: Raw string response from the LLM.
            query: Original query string.
            context_pages: Pages that were sent as context.

        Returns:
            Parsed GeneratedAnswer object.
        """
        # Try strict JSON parse
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed — attempting regex extraction")
            data = self._regex_extract(raw_response)

        answer_text = data.get("answer", "Unable to parse answer from model response.")
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Parse citations
        raw_citations = data.get("citations", [])
        citations: List[Citation] = []
        valid_page_ids = {p.page_id for p in context_pages}

        for c in raw_citations:
            if not isinstance(c, dict):
                continue
            page_id = c.get("page_id", "")
            # Only include citations that reference pages actually in context
            if page_id not in valid_page_ids:
                logger.warning(
                    "LLM cited page_id=%r not in context — keeping anyway", page_id
                )
            citations.append(
                Citation(
                    page_id=page_id,
                    company=c.get("company", ""),
                    year=int(c.get("year", 0)),
                    page_number=int(c.get("page_number", 0)),
                    relevance_note=c.get("relevance_note", ""),
                )
            )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            query=query,
            llm_backend=f"{self.backend}/{self.model_name}",
            raw_response=raw_response,
        )

    @staticmethod
    def _regex_extract(raw: str) -> dict:
        """Fallback: extract JSON from raw LLM response using regex.

        Args:
            raw: Raw response string possibly containing JSON.

        Returns:
            Parsed dict or minimal fallback dict.
        """
        # Try to find a JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: extract answer text
        answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
        return {
            "answer": answer_match.group(1) if answer_match else raw[:500],
            "citations": [],
            "confidence": float(confidence_match.group(1)) if confidence_match else 0.3,
        }

    # ------------------------------------------------------------------
    # Prompt Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(query: str, pages: List) -> str:
        """Build the user-facing prompt with page context summary.

        Args:
            query: The user query.
            pages: List of RetrievedPage objects in context.

        Returns:
            Formatted user prompt string.
        """
        page_summary = ", ".join(
            f"{p.company} {p.year} (p.{p.page_number})" for p in pages
        )
        return USER_PROMPT_TEMPLATE.format(
            query=query,
            num_pages=len(pages),
            page_summary=page_summary,
        )

    # ------------------------------------------------------------------
    # Initialization Helpers
    # ------------------------------------------------------------------

    def _init_gemini(self) -> None:
        """Initialize the Gemini SDK with the API key from environment."""
        if self._client is not None:
            return

        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Add it to your .env file."
            )
        genai.configure(api_key=api_key)
        self._client = genai
        logger.info("Gemini SDK initialized")

    def _init_openai(self) -> None:
        """Initialize the OpenAI SDK with the API key from environment."""
        if self._client is not None:
            return

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        self._client = OpenAI(api_key=api_key)
        logger.info("OpenAI SDK initialized")

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image to a base64-encoded PNG string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64-encoded string (no data URI prefix).
        """
        import base64

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
