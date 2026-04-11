"""Tool definitions, implementations, and web scraping helpers."""

from __future__ import annotations

import json
import os
import re

import requests
from ddgs import DDGS

# ── Tool definitions (OpenAI-compatible format) ─────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract text content from a URL (e.g., a job posting page)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information about a company, role, or interview topics",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_session",
            "description": "Save interview session summary or notes to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "File path to save to"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
]

# ── Tool implementations ────────────────────────────────────────────────────


def _fetch_with_browser(url: str) -> str | None:
    """Fetch a URL using a headless browser (Playwright). Returns HTML or None on failure."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                )
            )
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            # Wait a bit for JS to render job content
            page.wait_for_timeout(3000)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


def fetch_url(url: str) -> str:
    """Fetch a URL and extract job description text.

    Handles JS-heavy sites like Indeed by:
    1. Checking for structured JSON-LD job posting data embedded in the page
    2. Falling back to BeautifulSoup text extraction from common job-content selectors
    3. If the page yields very little text (JS-rendered), searching DuckDuckGo for
       the job title + company as a last resort.
    """
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    session = requests.Session()
    try:
        resp = session.get(url, timeout=15, headers=headers, allow_redirects=True)
    except Exception as e:
        return f"Error fetching URL: {e}"

    # If blocked (403/429), try headless browser before falling back to search
    used_browser = False
    if resp.status_code in (403, 429):
        browser_html = _fetch_with_browser(url)
        if browser_html:
            soup = BeautifulSoup(browser_html, "html.parser")
            used_browser = True
        else:
            # Browser failed too — fall back to web search
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            job_query = params.get("q", [""])[0]
            location = params.get("l", [""])[0]
            job_key = params.get("jk", [""])[0]
            if job_query and location:
                query = f"{job_query} job opening {location}"
            elif job_query:
                query = f"{job_query} job posting"
            elif job_key:
                query = f"indeed {job_key}"
            else:
                query = re.sub(r"https?://(www\.)?", "", url).split("?")[0]
            try:
                results = DDGS().text(query, max_results=5)
                if results:
                    parts = []
                    for r in results:
                        parts.append(f"**{r['title']}**\n{r['body']}\n{r['href']}")
                    search_text = "\n\n".join(parts)
                else:
                    search_text = "No results found."
            except Exception as e:
                search_text = f"Search error: {e}"
            return (
                f"[Could not access page. Searching for the job posting instead.]\n\n"
                + search_text
            )

    if resp.status_code >= 400 and not used_browser:
        return f"Error fetching URL: HTTP {resp.status_code}"

    if not used_browser:
        soup = BeautifulSoup(resp.text, "html.parser")

    # Strategy 1: JSON-LD structured data (Indeed, LinkedIn, many job boards embed this)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            # Handle both single objects and arrays
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") == "JobPosting":
                    parts = []
                    if item.get("title"):
                        parts.append(f"Title: {item['title']}")
                    if item.get("hiringOrganization", {}).get("name"):
                        parts.append(f"Company: {item['hiringOrganization']['name']}")
                    if item.get("jobLocation"):
                        loc = item["jobLocation"]
                        if isinstance(loc, dict):
                            addr = loc.get("address", {})
                            parts.append(f"Location: {addr.get('addressLocality', '')} {addr.get('addressRegion', '')}")
                    desc = item.get("description", "")
                    # Description is often HTML
                    if desc:
                        desc_soup = BeautifulSoup(desc, "html.parser")
                        parts.append(f"\n{desc_soup.get_text(separator='\n', strip=True)}")
                    if parts:
                        return "\n".join(parts)[:5000]
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

    # Strategy 2: Extract text from common job description selectors
    selectors = [
        {"id": "jobDescriptionText"},          # Indeed
        {"class_": "jobsearch-jobDescriptionText"},  # Indeed alt
        {"class_": "description__text"},        # LinkedIn
        {"class_": "job-description"},          # Generic
        {"class_": "posting-requirements"},     # Lever
        {"id": "job-details"},                  # Generic
    ]
    for sel in selectors:
        el = soup.find("div", **sel) or soup.find("section", **sel)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 100:
                return text[:5000]

    # Strategy 3: Full page text extraction (strip nav, script, style)
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # If we got meaningful text, return it
    if len(text) > 200:
        return text[:5000]

    # Strategy 4: Page was mostly JS-rendered, fall back to web search
    title_tag = soup.find("title")
    query = title_tag.get_text(strip=True) if title_tag else "job posting"
    return f"[Page content was JavaScript-rendered and could not be extracted. Searching the web instead.]\n\n{web_search(query + ' job description')}"


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(f"**{r['title']}**\n{r['body']}\n{r['href']}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search error: {e}"


def save_session(filepath: str, content: str) -> str:
    """Save content to a file on the server."""
    try:
        filepath = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return f"Session saved to {filepath}"
    except Exception as e:
        return f"Error saving: {e}"


_TOOL_FUNCTIONS = {
    "fetch_url": fetch_url,
    "web_search": web_search,
    "save_session": save_session,
}


def execute_tool(tool_call) -> dict:
    """Run a single tool call and return the result as a tool message."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    func = _TOOL_FUNCTIONS.get(name)
    result = func(**args) if func else f"Unknown tool: {name}"
    return {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
