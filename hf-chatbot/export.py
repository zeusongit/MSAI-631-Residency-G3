"""PDF export of chat sessions."""

from __future__ import annotations

import tempfile


def export_chat_pdf(history: list[dict]) -> str | None:
    """Export the chat history as a PDF file and return the file path."""
    if not history:
        return None
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Interview Prep Session", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)

    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        content = item.get("content", "")
        if role not in ("user", "assistant") or not content:
            continue

        label = "You" if role == "user" else "Coach"
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"{label}:", ln=True)
        pdf.set_font("Helvetica", size=10)
        # Encode to latin-1 for fpdf, replacing unsupported chars
        safe_text = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_text)
        pdf.ln(3)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    return tmp.name
