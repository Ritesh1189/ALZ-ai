"""
Module: resume_parser.py
Handles parsing and extracting textual content from PDF and DOCX documents
in-memory without persistent local storage, ensuring privacy and compliance.
"""

import io
from typing import Dict, Any, Tuple
from pypdf import PdfReader
from docx import Document
from text_cleaner import clean_text, extract_sections


MAX_ALLOWED_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def validate_file(file_name: str, file_size: int) -> Tuple[bool, str]:
    """
    Validates uploaded file extension and maximum file size.
    """
    if file_size > MAX_ALLOWED_FILE_SIZE_BYTES:
        return False, f"File size exceeds the 10 MB limit ({file_size / (1024 * 1024):.2f} MB)."

    valid_extensions = (".pdf", ".docx", ".txt")
    lower_name = file_name.lower()
    if not any(lower_name.endswith(ext) for ext in valid_extensions):
        return False, "Unsupported file format. Please upload a PDF, DOCX, or TXT resume."

    return True, "File is valid."


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """
    Extracts text from all pages of a PDF document using pypdf.
    Returns (extracted_text, total_pages).
    """
    pdf_stream = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)
    total_pages = len(reader.pages)

    extracted_pages = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            extracted_pages.append(page_text.strip())

    full_text = "\n\n".join(extracted_pages)
    return full_text, total_pages


def extract_text_from_docx(file_bytes: bytes) -> Tuple[str, int]:
    """
    Extracts text from paragraphs and tables of a DOCX document.
    Returns (extracted_text, paragraph_count).
    """
    docx_stream = io.BytesIO(file_bytes)
    doc = Document(docx_stream)

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Also capture table contents if present
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    full_text = "\n".join(paragraphs)
    return full_text, len(paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> Tuple[str, int]:
    """
    Extracts text from a plain text resume handling UTF-8 with BOM, UTF-16,
    CP1252, and Latin-1 encodings reliably.
    """
    text = ""
    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            candidate = file_bytes.decode(enc)
            # Ensure decoding does not produce null bytes from wrong endianness
            if "\x00" not in candidate:
                text = candidate
                break
        except (UnicodeDecodeError, LookupError):
            continue

    if not text:
        text = file_bytes.decode("utf-8", errors="replace")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [l for l in text.split("\n") if l.strip()]
    return text, len(lines)



def parse_resume(file_name: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Main entry point for parsing resume file bytes.
    Extracts raw text, cleans it, segments it, and gathers metadata.
    """
    lower_name = file_name.lower()
    element_count = 0

    if lower_name.endswith(".pdf"):
        raw_text, element_count = extract_text_from_pdf(file_bytes)
        source_format = "PDF"
    elif lower_name.endswith(".docx"):
        raw_text, element_count = extract_text_from_docx(file_bytes)
        source_format = "DOCX"
    elif lower_name.endswith(".txt"):
        raw_text, element_count = extract_text_from_txt(file_bytes)
        source_format = "TXT"
    else:
        raise ValueError(f"Unsupported file type: {file_name}")

    if not raw_text.strip():
        raise ValueError("The uploaded document contains no readable text.")

    cleaned = clean_text(raw_text)
    sections = extract_sections(raw_text)

    return {
        "file_name": file_name,
        "format": source_format,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "sections": sections,
        "element_count": element_count,
        "word_count": len(cleaned.split()),
        "character_count": len(raw_text),
    }
