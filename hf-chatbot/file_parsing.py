"""File text extraction for uploaded documents."""

from __future__ import annotations

import os


def extract_file_text(file_path: str) -> str:
    """Extract text from an uploaded PDF, DOCX, or plain text file."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text[:5000]
        elif ext in (".docx", ".doc"):
            import docx
            doc = docx.Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
            return text[:5000]
        else:
            with open(file_path) as f:
                return f.read()[:5000]
    except Exception as e:
        return f"Error reading file: {e}"
