"""Score parsing and formatting helpers."""

from __future__ import annotations

import re


def parse_score(text: str) -> tuple[str, int | None]:
    """Extract <!--SCORE:N--> from the end of a response. Returns (clean_text, score)."""
    match = re.search(r"<!--SCORE:(\d)-->", text)
    if match:
        score = int(match.group(1))
        clean = text[:match.start()].rstrip()
        return clean, max(1, min(5, score))
    return text, None


def fmt_avg(scores: list) -> str:
    """Format the average score for display."""
    if not scores:
        return ""
    avg = sum(scores) / len(scores)
    return f"{avg:.1f}/5 ({len(scores)} Qs)"


def format_duration(seconds: float) -> str:
    """Format seconds into a readable string like '1m 23s'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
