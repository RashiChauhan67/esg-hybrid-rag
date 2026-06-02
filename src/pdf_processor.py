"""
ESG RAG System — PDF Processor Module

Converts PDF ESG sustainability reports into per-page PIL images
and extracts OCR text for the BM25 retrieval index.

Uses PyMuPDF (fitz) for high-resolution image rendering and
pdfplumber for reliable text/table extraction.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PageImage:
    """Represents a single rendered page from a PDF document.

    Attributes:
        image: PIL Image object of the rendered page.
        page_id: Unique string identifier, e.g. "tesla_2023_p5".
        company: Company name extracted from the filename or metadata.
        year: Publication year parsed from the PDF or filename.
        page_number: 1-indexed page number within the document.
        pdf_path: Absolute path to the source PDF file.
        ocr_text: OCR-extracted plain text for BM25 indexing.
        image_path: Optional path where the image has been saved to disk.
    """

    image: Image.Image
    page_id: str
    company: str
    year: int
    page_number: int
    pdf_path: str
    ocr_text: str = ""
    image_path: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"PageImage(page_id={self.page_id!r}, company={self.company!r}, "
            f"year={self.year}, page_number={self.page_number})"
        )


class PDFProcessor:
    """Converts PDF ESG reports to per-page images with OCR text extraction.

    This class is the entry point for the ingestion pipeline. For each
    PDF provided, it renders every page at the specified DPI using
    PyMuPDF and extracts raw text via pdfplumber.

    Args:
        dpi: Dots-per-inch for image rendering. Higher = better quality
            but larger memory. Default 150 is suitable for ColPali.
        image_format: Output image format ("PNG" or "JPEG").
        max_pages: Optional maximum number of pages to process per PDF.
            Set to None (default) to process all pages.

    Example:
        >>> processor = PDFProcessor(dpi=150)
        >>> pages = processor.process_pdf("reports/tesla_2023.pdf")
        >>> print(pages[0].page_id)
        'tesla_2023_p1'
    """

    def __init__(
        self,
        dpi: int = 150,
        image_format: str = "PNG",
        max_pages: Optional[int] = None,
    ) -> None:
        self.dpi = dpi
        self.image_format = image_format
        self.max_pages = max_pages
        self._zoom = dpi / 72.0  # PyMuPDF native DPI is 72
        logger.info(
            "PDFProcessor initialized: dpi=%d, format=%s, max_pages=%s",
            dpi,
            image_format,
            max_pages,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str | Path) -> List[PageImage]:
        """Process a single PDF file into a list of PageImage objects.

        Renders each page at the configured DPI and extracts OCR text
        using pdfplumber. Metadata (company, year) is inferred from the
        filename and document content.

        Args:
            pdf_path: Path to the PDF file to process.

        Returns:
            Ordered list of PageImage objects, one per rendered page.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
            ValueError: If the PDF cannot be opened or parsed.
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Processing PDF: %s", pdf_path)
        company, year = self._infer_metadata(pdf_path)
        logger.debug("Inferred metadata — company: %s, year: %d", company, year)

        try:
            pages = self._render_pages(pdf_path, company, year)
        except Exception as exc:
            raise ValueError(f"Failed to process PDF '{pdf_path}': {exc}") from exc

        logger.info(
            "Extracted %d pages from %s (company=%s, year=%d)",
            len(pages),
            pdf_path.name,
            company,
            year,
        )
        return pages

    def process_directory(self, directory: str | Path) -> List[PageImage]:
        """Process all PDF files in a directory.

        Args:
            directory: Path to a directory containing PDF files.

        Returns:
            Flat list of PageImage objects from all PDFs found.

        Raises:
            FileNotFoundError: If ``directory`` does not exist.
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pdf_files = sorted(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in: %s", directory)
            return []

        logger.info("Found %d PDF files in %s", len(pdf_files), directory)
        all_pages: List[PageImage] = []
        for pdf_file in pdf_files:
            try:
                pages = self.process_pdf(pdf_file)
                all_pages.extend(pages)
            except Exception as exc:
                logger.error("Skipping %s due to error: %s", pdf_file.name, exc)

        logger.info("Total pages extracted: %d", len(all_pages))
        return all_pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render_pages(
        self, pdf_path: Path, company: str, year: int
    ) -> List[PageImage]:
        """Render all pages of a PDF to PIL Images using PyMuPDF.

        Args:
            pdf_path: Resolved path to the PDF file.
            company: Company name for metadata.
            year: Year for metadata.

        Returns:
            List of PageImage objects.
        """
        pages: List[PageImage] = []
        matrix = fitz.Matrix(self._zoom, self._zoom)

        # Extract text with pdfplumber in one pass for efficiency
        text_by_page = self._extract_text_pdfplumber(pdf_path)

        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            limit = (
                min(self.max_pages, total_pages)
                if self.max_pages
                else total_pages
            )
            logger.debug("Rendering %d of %d pages", limit, total_pages)

            for page_idx in range(limit):
                fitz_page = doc[page_idx]
                page_number = page_idx + 1

                # Render to pixmap then convert to PIL Image
                pixmap = fitz_page.get_pixmap(matrix=matrix, alpha=False)
                pil_image = Image.frombytes(
                    "RGB",
                    [pixmap.width, pixmap.height],
                    pixmap.samples,
                )

                page_id = f"{company}_{year}_p{page_number}"
                ocr_text = text_by_page.get(page_idx, "")

                page = PageImage(
                    image=pil_image,
                    page_id=page_id,
                    company=company,
                    year=year,
                    page_number=page_number,
                    pdf_path=str(pdf_path),
                    ocr_text=ocr_text,
                )
                pages.append(page)
                logger.debug("Rendered page %d/%d: %s", page_number, limit, page_id)

        return pages

    def _extract_text_pdfplumber(self, pdf_path: Path) -> dict[int, str]:
        """Extract text from all pages using pdfplumber.

        Falls back gracefully if pdfplumber cannot parse certain pages.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict mapping 0-indexed page number to extracted text string.
        """
        text_map: dict[int, str] = {}
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    if self.max_pages and i >= self.max_pages:
                        break
                    try:
                        text = page.extract_text() or ""
                        # Also extract table content as text
                        tables = page.extract_tables() or []
                        table_text = self._tables_to_text(tables)
                        text_map[i] = f"{text}\n{table_text}".strip()
                    except Exception as exc:
                        logger.warning(
                            "pdfplumber failed on page %d: %s", i + 1, exc
                        )
                        text_map[i] = ""
        except Exception as exc:
            logger.error("pdfplumber could not open %s: %s", pdf_path, exc)
        return text_map

    @staticmethod
    def _tables_to_text(tables: list) -> str:
        """Convert nested table lists from pdfplumber to plain text.

        Args:
            tables: List of tables, each a list of row lists.

        Returns:
            Concatenated text representation of all tables.
        """
        lines: List[str] = []
        for table in tables:
            for row in table:
                if row:
                    row_text = " | ".join(
                        str(cell).strip() if cell else "" for cell in row
                    )
                    lines.append(row_text)
        return "\n".join(lines)

    def _infer_metadata(self, pdf_path: Path) -> tuple[str, int]:
        """Infer company name and year from the PDF filename.

        Supports filenames like:
            - "tesla_2023.pdf"           → ("tesla", 2023)
            - "MSFT_ESG_Report_2022.pdf" → ("msft", 2022)
            - "sustainability2024.pdf"   → ("sustainability", 2024)
            - "report.pdf"               → ("report", 2024)

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Tuple of (company_slug, year).
        """
        stem = pdf_path.stem.lower()
        # Try to find a 4-digit year
        year_match = re.search(r"(20\d{2}|19\d{2})", stem)
        year = int(year_match.group(1)) if year_match else 2024

        # Remove year and common ESG report keywords to get company name
        clean = re.sub(r"(20\d{2}|19\d{2})", "", stem)
        clean = re.sub(r"[_\-\s]+(esg|report|sustainability|csr|annual|gri)+", "", clean)
        clean = re.sub(r"[_\-\s]+", "_", clean).strip("_")
        company = clean if clean else "unknown"

        return company, year
