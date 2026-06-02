"""
ESG RAG System — Corpus Builder Module

Builds and manages the visual corpus index from multiple ESG PDF reports.
Serializes page images and metadata to disk for reproducibility and
efficient reloading without reprocessing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .pdf_processor import PDFProcessor, PageImage

logger = logging.getLogger(__name__)


@dataclass
class CorpusMetadata:
    """Metadata for the entire visual corpus.

    Attributes:
        total_pages: Total number of pages across all documents.
        documents: List of document-level metadata dicts.
        corpus_dir: Absolute path to the corpus storage directory.
        dpi: DPI used for rendering (for reproducibility).
        version: Schema version for forward compatibility.
    """

    total_pages: int
    documents: List[Dict] = field(default_factory=list)
    corpus_dir: str = ""
    dpi: int = 150
    version: str = "1.0"


class CorpusBuilder:
    """Builds and manages a serialized visual corpus from ESG PDF reports.

    The corpus is stored on disk as:
        <corpus_dir>/
            images/
                <page_id>.png       # Rendered page images
            corpus_metadata.json    # Full page metadata + OCR text

    Supports incremental indexing: PDFs already in the corpus are skipped
    on subsequent runs (unless force_reindex=True).

    Args:
        corpus_dir: Directory to store the corpus.
        dpi: DPI for PDF rendering (passed to PDFProcessor).
        max_pages: Max pages per PDF (None = all pages).
        force_reindex: If True, re-processes all PDFs even if cached.

    Example:
        >>> builder = CorpusBuilder("./data/corpus")
        >>> pages = builder.build(["reports/tesla_2023.pdf", "reports/msft_2022.pdf"])
        >>> print(f"Indexed {len(pages)} pages")
    """

    METADATA_FILENAME = "corpus_metadata.json"

    def __init__(
        self,
        corpus_dir: str | Path,
        dpi: int = 150,
        max_pages: Optional[int] = None,
        force_reindex: bool = False,
    ) -> None:
        self.corpus_dir = Path(corpus_dir).resolve()
        self.images_dir = self.corpus_dir / "images"
        self.dpi = dpi
        self.max_pages = max_pages
        self.force_reindex = force_reindex

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._processor = PDFProcessor(dpi=dpi, max_pages=max_pages)
        logger.info("CorpusBuilder initialized at: %s", self.corpus_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, pdf_paths: List[str | Path]) -> List[PageImage]:
        """Build corpus from a list of PDF paths.

        Processes each PDF, saves images and metadata to disk, and
        returns the full list of PageImage objects with image_path set.

        Args:
            pdf_paths: List of paths to PDF files to index.

        Returns:
            Full list of PageImage objects in the corpus.

        Raises:
            ValueError: If no valid PDF paths are provided.
        """
        if not pdf_paths:
            raise ValueError("No PDF paths provided to CorpusBuilder.build()")

        existing_ids = self._load_existing_ids()
        all_pages: List[PageImage] = []

        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            logger.info("Processing PDF: %s", pdf_path.name)

            try:
                new_pages = self._processor.process_pdf(pdf_path)
            except Exception as exc:
                logger.error("Failed to process %s: %s", pdf_path.name, exc)
                continue

            for page in new_pages:
                if page.page_id in existing_ids and not self.force_reindex:
                    logger.debug("Skipping already-indexed page: %s", page.page_id)
                    continue
                page = self._save_page(page)
                all_pages.append(page)

        if all_pages:
            self._append_metadata(all_pages)
            logger.info(
                "Corpus updated: %d new pages added to %s",
                len(all_pages),
                self.corpus_dir,
            )

        return self.load_corpus()

    def load_corpus(self) -> List[PageImage]:
        """Load all pages from the saved corpus metadata.

        Reconstructs PageImage objects from disk. Images are loaded
        lazily (only if the image_path exists on disk).

        Returns:
            List of PageImage objects from the stored corpus.

        Raises:
            FileNotFoundError: If corpus has not been built yet.
        """
        metadata_path = self.corpus_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Corpus metadata not found at {metadata_path}. "
                "Run CorpusBuilder.build() first."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        pages: List[PageImage] = []
        for record in records:
            image_path = Path(record["image_path"])
            if not image_path.exists():
                logger.warning("Image file missing: %s — skipping", image_path)
                continue

            image = Image.open(image_path).convert("RGB")
            page = PageImage(
                image=image,
                page_id=record["page_id"],
                company=record["company"],
                year=record["year"],
                page_number=record["page_number"],
                pdf_path=record["pdf_path"],
                ocr_text=record.get("ocr_text", ""),
                image_path=str(image_path),
            )
            pages.append(page)

        logger.info("Loaded %d pages from corpus at %s", len(pages), self.corpus_dir)
        return pages

    def get_corpus_stats(self) -> Dict:
        """Return summary statistics about the current corpus.

        Returns:
            Dict with keys: total_pages, companies, years, corpus_dir.
        """
        metadata_path = self.corpus_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            return {"total_pages": 0, "companies": [], "years": [], "corpus_dir": str(self.corpus_dir)}

        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        companies = sorted(set(r["company"] for r in records))
        years = sorted(set(r["year"] for r in records))
        return {
            "total_pages": len(records),
            "companies": companies,
            "years": years,
            "corpus_dir": str(self.corpus_dir),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_page(self, page: PageImage) -> PageImage:
        """Save a page image to disk and return the updated PageImage.

        Args:
            page: PageImage with an in-memory PIL image.

        Returns:
            PageImage with image_path set to the saved file path.
        """
        filename = f"{page.page_id}.png"
        image_path = self.images_dir / filename
        page.image.save(str(image_path), format="PNG", optimize=False)
        page.image_path = str(image_path)
        logger.debug("Saved page image: %s", image_path)
        return page

    def _load_existing_ids(self) -> set[str]:
        """Load set of already-indexed page IDs from the metadata file.

        Returns:
            Set of page_id strings already in the corpus.
        """
        metadata_path = self.corpus_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            return set()
        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {r["page_id"] for r in records}

    def _append_metadata(self, new_pages: List[PageImage]) -> None:
        """Append new page metadata to the corpus metadata JSON.

        Args:
            new_pages: List of newly processed PageImage objects.
        """
        metadata_path = self.corpus_dir / self.METADATA_FILENAME

        # Load existing records
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = []

        # Build set of existing IDs for dedup
        existing_ids = {r["page_id"] for r in records}

        for page in new_pages:
            if page.page_id in existing_ids:
                continue
            records.append(
                {
                    "page_id": page.page_id,
                    "company": page.company,
                    "year": page.year,
                    "page_number": page.page_number,
                    "pdf_path": page.pdf_path,
                    "ocr_text": page.ocr_text,
                    "image_path": page.image_path or "",
                }
            )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(
            "Metadata updated: %d total records in %s",
            len(records),
            metadata_path,
        )
