"""
ESG RAG System — Benchmark Dataset Builder

Automatically generates Q&A benchmark datasets from ESG PDF reports
using a multimodal LLM. For each page or set of pages, the LLM is
asked to generate questions that require reading that specific page
to answer — creating a visually-grounded evaluation benchmark.

The output benchmark.json format matches the evaluation pipeline:
[
    {
        "query": str,
        "relevant_pages": [page_id, ...],
        "answer": str,
        "company": str,
        "year": int,
        "difficulty": "easy" | "medium" | "hard"
    }
]
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """A single benchmark Q&A entry for ESG retrieval evaluation.

    Attributes:
        query: The question to be answered.
        relevant_pages: List of page_ids that contain the answer.
        answer: Reference answer string.
        company: Company the question is about.
        year: Report year.
        difficulty: Estimated difficulty ("easy", "medium", "hard").
        source_context: What visual element prompted the question
            (e.g., "emissions table", "chart on p.12").
    """

    query: str
    relevant_pages: List[str]
    answer: str
    company: str
    year: int
    difficulty: str = "medium"
    source_context: str = ""

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "query": self.query,
            "relevant_pages": self.relevant_pages,
            "answer": self.answer,
            "company": self.company,
            "year": self.year,
            "difficulty": self.difficulty,
            "source_context": self.source_context,
        }


# ======================================================================
# Benchmark Generation Prompts
# ======================================================================

BENCHMARK_SYSTEM_PROMPT = """You are an expert ESG (Environmental, Social, 
and Governance) research analyst creating evaluation questions for a 
visual document retrieval system.

You will be shown a page from an ESG sustainability report. Your task is to 
generate exactly {num_questions} questions that:
1. REQUIRE reading this specific page to answer (cannot be answered from general knowledge)
2. Are based on specific numbers, charts, tables, or text visible on the page
3. Have definitive, verifiable answers directly from the page content
4. Vary in difficulty (include easy/factual and harder/analytical questions)

RESPONSE FORMAT — Respond with a JSON array only:
[
    {
        "query": "What is [Company]'s Scope 1 GHG emissions for 2022?",
        "answer": "X metric tons CO2 equivalent, as shown in the emissions table",
        "difficulty": "easy",
        "source_context": "Scope 1 emissions table in the middle of the page"
    },
    ...
]

QUESTION TYPES TO INCLUDE:
- Factual (specific numbers, dates, percentages from tables/charts)  
- Comparative (year-over-year changes, benchmark comparisons)
- Target/Goal (commitments and future targets stated on the page)
- Trend (directional trends visible in charts)
"""

BENCHMARK_USER_PROMPT = """Page: {page_id}
Company: {company}
Year: {year}
Page Number: {page_number}

Please analyze this ESG report page and generate {num_questions} evaluation questions 
in the required JSON format. Make sure each question can ONLY be answered by looking 
at this specific page."""


class BenchmarkBuilder:
    """Generates Q&A benchmark datasets from ESG PDF pages using an LLM.

    Uses a multimodal LLM (Gemini or GPT-4o) to generate questions that
    require visual comprehension of specific ESG report pages. This creates
    a grounded evaluation benchmark tied to actual document content.

    Args:
        backend: LLM backend ("gemini" or "openai").
        model_name: Specific model name. Defaults to backend default.
        questions_per_page: Number of Q&A pairs to generate per page.
        max_pages_per_doc: Maximum pages to sample per document. Set to
            None to process all pages.
        delay_between_calls: Seconds to wait between API calls to avoid
            rate limiting.

    Example:
        >>> builder = BenchmarkBuilder(backend="gemini", questions_per_page=3)
        >>> entries = builder.generate_from_pages(pages, max_total=50)
        >>> builder.save(entries, "data/benchmark.json")
    """

    def __init__(
        self,
        backend: str = "gemini",
        model_name: Optional[str] = None,
        questions_per_page: int = 5,
        max_pages_per_doc: Optional[int] = 10,
        delay_between_calls: float = 1.0,
    ) -> None:
        self.backend = backend.lower()
        self.questions_per_page = questions_per_page
        self.max_pages_per_doc = max_pages_per_doc
        self.delay_between_calls = delay_between_calls

        if model_name:
            self.model_name = model_name
        elif self.backend == "gemini":
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

        self._client = None
        logger.info(
            "BenchmarkBuilder initialized: backend=%s, model=%s, "
            "questions_per_page=%d",
            backend,
            self.model_name,
            questions_per_page,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_pages(
        self,
        pages: List,
        max_total: Optional[int] = None,
        sample_strategy: str = "uniform",
    ) -> List[BenchmarkEntry]:
        """Generate benchmark entries from a list of PageImage objects.

        Args:
            pages: List of PageImage objects to generate questions from.
            max_total: Maximum total Q&A entries to generate. None = unlimited.
            sample_strategy: "uniform" (evenly sample across docs) or
                "all" (process all provided pages).

        Returns:
            List of BenchmarkEntry objects.

        Raises:
            ValueError: If pages list is empty.
        """
        if not pages:
            raise ValueError("pages list must not be empty.")

        # Sample pages if needed
        selected_pages = self._select_pages(pages, sample_strategy)
        logger.info(
            "Generating benchmark from %d/%d pages (%s strategy)",
            len(selected_pages),
            len(pages),
            sample_strategy,
        )

        all_entries: List[BenchmarkEntry] = []

        for i, page in enumerate(selected_pages):
            if max_total and len(all_entries) >= max_total:
                logger.info("Reached max_total=%d entries — stopping", max_total)
                break

            logger.info(
                "Processing page %d/%d: %s", i + 1, len(selected_pages), page.page_id
            )

            try:
                entries = self._generate_for_page(page)
                all_entries.extend(entries)
                logger.info(
                    "Generated %d entries for %s (total: %d)",
                    len(entries),
                    page.page_id,
                    len(all_entries),
                )
            except Exception as exc:
                logger.error(
                    "Failed to generate for page %s: %s", page.page_id, exc
                )

            if i < len(selected_pages) - 1:
                time.sleep(self.delay_between_calls)

        logger.info(
            "Benchmark generation complete: %d total entries", len(all_entries)
        )
        return all_entries

    def save(
        self,
        entries: List[BenchmarkEntry],
        output_path: str | Path,
        indent: int = 2,
    ) -> None:
        """Save benchmark entries to a JSON file.

        Args:
            entries: List of BenchmarkEntry objects.
            output_path: Path to the output JSON file.
            indent: JSON indentation level.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [e.to_dict() for e in entries]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.info(
            "Saved %d benchmark entries to %s", len(entries), output_path
        )

    @staticmethod
    def load(benchmark_path: str | Path) -> List[BenchmarkEntry]:
        """Load benchmark entries from a JSON file.

        Args:
            benchmark_path: Path to the benchmark JSON file.

        Returns:
            List of BenchmarkEntry objects.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(benchmark_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = []
        for record in data:
            entries.append(
                BenchmarkEntry(
                    query=record["query"],
                    relevant_pages=record["relevant_pages"],
                    answer=record["answer"],
                    company=record.get("company", ""),
                    year=record.get("year", 2024),
                    difficulty=record.get("difficulty", "medium"),
                    source_context=record.get("source_context", ""),
                )
            )

        logger.info("Loaded %d benchmark entries from %s", len(entries), path)
        return entries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_for_page(self, page) -> List[BenchmarkEntry]:
        """Generate Q&A entries for a single page using the LLM.

        Args:
            page: PageImage object.

        Returns:
            List of BenchmarkEntry objects for this page.
        """
        system_prompt = BENCHMARK_SYSTEM_PROMPT.format(
            num_questions=self.questions_per_page
        )
        user_prompt = BENCHMARK_USER_PROMPT.format(
            page_id=page.page_id,
            company=page.company,
            year=page.year,
            page_number=page.page_number,
            num_questions=self.questions_per_page,
        )

        if self.backend == "gemini":
            raw = self._call_gemini(system_prompt, user_prompt, page.image)
        else:
            raw = self._call_openai(system_prompt, user_prompt, page.image)

        return self._parse_qa_response(raw, page)

    def _call_gemini(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        """Call Gemini with a page image to generate Q&A pairs."""
        self._init_gemini()

        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
        response = model.generate_content([user_prompt, image])
        return response.text

    def _call_openai(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        """Call OpenAI GPT-4o with a page image to generate Q&A pairs."""
        import base64
        import io

        self._init_openai()

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64_image = base64.b64encode(buffer.getvalue()).decode()

        content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            },
        ]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.4,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_qa_response(self, raw: str, page) -> List[BenchmarkEntry]:
        """Parse LLM response into BenchmarkEntry objects.

        Args:
            raw: Raw JSON string from the LLM.
            page: Source PageImage object.

        Returns:
            List of BenchmarkEntry objects.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON for page %s", page.page_id)
                    return []
            else:
                logger.warning("No JSON array found in response for %s", page.page_id)
                return []

        # Handle {"questions": [...]} wrapper
        if isinstance(data, dict):
            data = data.get("questions", data.get("entries", [data]))

        entries: List[BenchmarkEntry] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            query = item.get("query", "").strip()
            answer = item.get("answer", "").strip()
            if not query or not answer:
                continue

            entries.append(
                BenchmarkEntry(
                    query=query,
                    relevant_pages=[page.page_id],
                    answer=answer,
                    company=page.company,
                    year=page.year,
                    difficulty=item.get("difficulty", "medium"),
                    source_context=item.get("source_context", ""),
                )
            )

        return entries

    def _select_pages(self, pages: List, strategy: str) -> List:
        """Select pages for benchmark generation.

        Args:
            pages: All available pages.
            strategy: "uniform" or "all".

        Returns:
            Selected subset of pages.
        """
        if strategy == "all" or self.max_pages_per_doc is None:
            return pages

        # Group by company and sample uniformly
        from collections import defaultdict
        import random

        by_company: Dict[str, List] = defaultdict(list)
        for page in pages:
            by_company[page.company].append(page)

        selected: List = []
        for company, company_pages in by_company.items():
            limit = min(self.max_pages_per_doc, len(company_pages))
            # Sample uniformly across the document (not just the start)
            step = max(1, len(company_pages) // limit)
            sampled = company_pages[::step][:limit]
            selected.extend(sampled)
            logger.debug(
                "Selected %d/%d pages from %s", len(sampled), len(company_pages), company
            )

        return selected

    def _init_gemini(self) -> None:
        """Initialize Gemini SDK."""
        if self._client is not None:
            return
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment.")
        genai.configure(api_key=api_key)
        self._client = genai

    def _init_openai(self) -> None:
        """Initialize OpenAI SDK."""
        if self._client is not None:
            return
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self._client = OpenAI(api_key=api_key)
