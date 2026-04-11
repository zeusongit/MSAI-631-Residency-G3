"""
Interview Prep Chatbot — Gradio UI powered by Hugging Face Inference API.

Given a job description (pasted text, URL, or uploaded file), the bot asks
tailored technical and behavioral interview questions, evaluates answers
with constructive feedback, and asks follow-up questions before moving on.

Supports voice input via browser microphone (transcribed with Whisper),
file upload (PDF/DOCX/TXT), difficulty selection, answer timing, session
export as PDF, dynamic UI (MCQ radio buttons, rating sliders), streaming
responses, per-answer scoring, resume upload, category focus, and a
"show model answer" button.

Set HUGGING_FACE_HUB_TOKEN for API access. See README for all env vars.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time

import gradio as gr
import numpy as np
import requests
from duckduckgo_search import DDGS
from faster_whisper import WhisperModel
from huggingface_hub import InferenceClient

# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

MODEL_ID = os.environ.get("HF_MODEL_ID", DEFAULT_MODEL_ID)
MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", "1024"))
WHISPER_MODEL_SIZE = os.environ.get("HF_WHISPER_MODEL", "small")
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "") or os.environ.get("HF_TOKEN", "")

DIFFICULTY_PROMPTS = {
    "Easy": (
        "Adjust your questions to an ENTRY-LEVEL / JUNIOR difficulty. "
        "Ask straightforward questions about fundamentals, basic concepts, "
        "and simple scenario-based technical questions. Be extra supportive."
    ),
    "Medium": (
        "Adjust your questions to a MID-LEVEL difficulty. "
        "Ask questions that require solid understanding, some depth, "
        "and real-world examples. Balance challenge with encouragement."
    ),
    "Hard": (
        "Adjust your questions to a SENIOR / STAFF-LEVEL difficulty. "
        "Ask deep technical questions, complex system design scenarios, "
        "and questions that probe system design trade-offs and technical leadership decisions. "
        "Be rigorous — this person wants to be pushed."
    ),
}

CATEGORY_PROMPTS = {
    "System Design": "Focus questions on system design: scalability, distributed systems, load balancing, caching, database choices, and architecture trade-offs.",
    "Coding/Algorithms": "Focus questions on coding and algorithms: data structures, time/space complexity, common algorithms, and ask the candidate to write or trace through code.",
    "Domain Knowledge": "Focus questions on domain-specific knowledge relevant to the job description: industry concepts, tools, frameworks, and best practices mentioned in the JD.",
    "API Design": "Focus questions on API design: REST vs GraphQL, endpoint design, versioning, authentication, rate limiting, error handling, and documentation.",
    "Debugging/Troubleshooting": "Focus questions on debugging and troubleshooting: reading error messages, diagnosing production issues, systematic debugging approaches, and monitoring/observability.",
    "Architecture": "Focus questions on software architecture: design patterns, microservices vs monoliths, event-driven architecture, CQRS, and architectural decision-making.",
}

SYSTEM_PROMPT = os.environ.get("HF_SYSTEM_PROMPT", """\
You are a technical interview coach. Your job is to drill candidates on the \
technical skills and knowledge required by the job description they provide. \
Keep responses concise and focused.

## Your style:
- Direct and professional — no small talk or filler
- Keep responses short (2-4 sentences typical)
- Focus on technical depth, not background or personality questions
- Give sharp, specific feedback on answers

## Flow:

**Getting started:** When the user shares a job description (text or URL — use \
fetch_url for URLs), read it carefully. Use web_search to research the company's \
tech stack if needed. Then immediately ask your first technical question — no \
preamble, no "tell me about yourself", no asking if they're ready.

**During the interview:** Ask ONE question at a time. Focus on:
- Technical concepts, system design, and problem-solving relevant to the role
- Hands-on scenarios ("How would you implement...", "Walk me through...")
- Trade-offs, edge cases, and debugging approaches
- Architecture and design decisions

After they answer:
- Point out what was correct or strong
- Identify what was missing or could be deeper
- Then ask the next technical question or a follow-up probing deeper

Do NOT number your questions. Do NOT ask basic background questions like \
"tell me about yourself" or "what interests you about this role." Stay technical.

**Wrapping up:** After 5-7 questions (don't count out loud), wrap up with a \
brief technical assessment — strongest areas, gaps to study, and specific \
topics to review for this role. Offer to save the summary.

## Important:
- NEVER ask conversational or background questions — stay technical
- NEVER list multiple questions at once
- NEVER use numbered steps or bullet-point feedback dumps
- If an answer is weak, probe deeper: "What about edge case X?" or "How would that change if..."
- If they say "skip" or "next", move to the next technical topic
- If they say "done" or "end", wrap up with the technical assessment

## Scoring:
After EVERY response where you evaluated the user's answer, append a single line at the \
very end of your message in this exact format:
<!--SCORE:N-->
where N is 1-5 (1=poor, 2=weak, 3=adequate, 4=strong, 5=excellent). \
Do NOT mention the score in your visible text. Only append this hidden tag. \
Do NOT include the score tag when you are asking the first question, wrapping up, \
or responding to commands like "skip" or "done".

## Code:
When discussing code, ALWAYS use markdown fenced code blocks with the language specified, \
e.g. ```python. When asking coding questions, encourage the candidate to write code in \
their response.

## Dynamic UI hints — FOLLOW EXACTLY:
After EVERY message that ends with a direct question to the user, append a single \
`<ui_hint>` tag on a new line at the very end (after the SCORE tag if present). \
Choose the type that best fits:

- Open-ended (default): <ui_hint>{"type": "open"}</ui_hint>
- Multiple choice — 4 distinct, plausible options: \
<ui_hint>{"type": "mcq", "options": ["Option A", "Option B", "Option C", "Option D"]}</ui_hint>
- Self-rating or confidence check: \
<ui_hint>{"type": "rating", "min": 1, "max": 5, "label": "How confident are you? (1=not at all, 5=very)"}</ui_hint>

Rules:
- ONLY append a hint when your message ends with a direct question
- Do NOT append a hint on pure feedback, acknowledgements, or summary messages
- For MCQ: generate realistic, clearly distinct options relevant to the question
- Use MCQ for factual/knowledge questions where a few clear answers exist
- Use rating for confidence checks, self-assessments, or "how well do you know X?"
- Use open for everything else (scenario, explanation, coding questions)
""")

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
    {
        "type": "function",
        "function": {
            "name": "search_resume",
            "description": (
                "Search the candidate's uploaded resume for relevant experience, skills, or projects. "
                "Call this before asking about any specific skill or domain to see what the candidate "
                "has listed. Examples: search_resume('Python projects'), "
                "search_resume('system design experience'), search_resume('leadership roles')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Skill, technology, or experience area to look up in the resume",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ── Tool implementations ────────────────────────────────────────────────────


def _fetch_url(url: str) -> str:
    """Fetch a URL and return its text content with HTML stripped."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
    except Exception as e:
        return f"Error fetching URL: {e}"


def _web_search(query: str) -> str:
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


def _save_session(filepath: str, content: str) -> str:
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
    "fetch_url": _fetch_url,
    "web_search": _web_search,
    "save_session": _save_session,
}


def _execute_tool(tool_call) -> dict:
    """Run a single tool call and return the result as a tool message."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    func = _TOOL_FUNCTIONS.get(name)
    result = func(**args) if func else f"Unknown tool: {name}"
    return {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}

# ── File parsing ────────────────────────────────────────────────────────────


def _extract_file_text(file_path: str) -> str:
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

# ── Whisper (voice transcription) ───────────────────────────────────────────

_whisper_model: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    """Lazy-load the Whisper model on first voice input."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, compute_type="int8")
    return _whisper_model


def _transcribe_audio(audio_tuple) -> str:
    """Transcribe audio from Gradio's Audio component (sample_rate, np_array)."""
    if audio_tuple is None:
        return ""
    sample_rate, audio_data = audio_tuple
    if audio_data is None or len(audio_data) == 0:
        return ""
    audio = audio_data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.max() > 1.0:
        audio = audio / 32768.0
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        tmp_path = tmp.name
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(tmp_path, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)

# ── RAG (Resume vector store) ────────────────────────────────────────────────

_rag_embedder = None
_rag_chroma_client = None
_rag_collection = None


def _get_rag_embedder():
    global _rag_embedder
    if _rag_embedder is None:
        from sentence_transformers import SentenceTransformer
        _rag_embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _rag_embedder


def _chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping fixed-size chunks for embedding."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def build_resume_index(resume_text: str) -> str:
    """Chunk the resume, embed it, and store in an in-memory ChromaDB collection."""
    global _rag_chroma_client, _rag_collection
    try:
        import chromadb
        if _rag_chroma_client is None:
            _rag_chroma_client = chromadb.Client()
        try:
            _rag_chroma_client.delete_collection("resume")
        except Exception:
            pass
        _rag_collection = _rag_chroma_client.create_collection("resume")
        chunks = _chunk_text(resume_text)
        if not chunks:
            return "Resume appears to be empty."
        embeddings = _get_rag_embedder().encode(chunks).tolist()
        _rag_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(chunks))],
        )
        return (
            f"Resume indexed ({len(chunks)} chunks). "
            "The coach will call search_resume before asking about specific skills."
        )
    except ImportError:
        return (
            "RAG dependencies not installed (sentence-transformers, chromadb). "
            "Resume will be passed as plain text instead."
        )
    except Exception as e:
        return f"Error indexing resume: {e}"


def _search_resume(query: str) -> str:
    """Retrieve the most relevant resume chunks for a given query."""
    if _rag_collection is None:
        return "No resume has been uploaded and indexed yet."
    try:
        n = min(3, _rag_collection.count())
        if n == 0:
            return "Resume index is empty."
        q_emb = _get_rag_embedder().encode([query]).tolist()
        results = _rag_collection.query(query_embeddings=q_emb, n_results=n)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No matching resume content found."
        return "\n---\n".join(docs)
    except Exception as e:
        return f"Resume search error: {e}"


# Register after definition to avoid forward-reference error
_TOOL_FUNCTIONS["search_resume"] = _search_resume

# ── Response tag parsing ─────────────────────────────────────────────────────

_UI_HINT_RE = re.compile(r"<ui_hint>(.*?)</ui_hint>", re.DOTALL)
_SCORE_RE = re.compile(r"<!--SCORE:(\d)-->")


def _parse_tags(text: str) -> tuple[str, dict, int | None]:
    """
    Strip <ui_hint> and <!--SCORE:N--> tags from model reply.
    Returns (clean_text, hint_dict, score_or_None).
    """
    # Parse score
    score: int | None = None
    score_match = _SCORE_RE.search(text)
    if score_match:
        score = max(1, min(5, int(score_match.group(1))))
        text = text[:score_match.start()].rstrip()

    # Parse ui_hint
    hint: dict = {"type": "open"}
    hint_match = _UI_HINT_RE.search(text)
    if hint_match:
        try:
            hint = json.loads(hint_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            hint = {"type": "open"}
        text = _UI_HINT_RE.sub("", text).strip()

    return text.strip(), hint, score


def _input_visibility(hint: dict) -> tuple:
    """
    Return gr.update() tuples for all dynamic input rows/components.
    Order: (text_row, mcq_row, mcq_radio, rating_row, rating_slider)

    The text input row is ALWAYS visible so the conversation can always
    continue. MCQ and rating rows appear above it as additional options.
    """
    qtype = hint.get("type", "open")
    if qtype == "mcq":
        options = hint.get("options") or []
        if options:
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(choices=options, value=None),
                gr.update(visible=False),
                gr.update(),
            )
    if qtype == "rating":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(visible=True),
            gr.update(
                minimum=hint.get("min", 1),
                maximum=hint.get("max", 5),
                value=hint.get("min", 1),
                label=hint.get("label", "Your rating"),
            ),
        )
    # default / open / mcq with no options: text input only
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        gr.update(visible=False),
        gr.update(),
    )


def _no_change_visibility() -> tuple:
    """gr.update() tuples that don't change any input row visibility."""
    return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

# ── Chat logic ──────────────────────────────────────────────────────────────


def _build_messages(
    history: list[dict],
    difficulty: str,
    categories: list[str] | None = None,
    resume_text: str = "",
) -> list[dict]:
    """Assemble the full message list from Gradio chat history."""
    messages: list[dict] = []
    prompt = SYSTEM_PROMPT.strip()
    if difficulty in DIFFICULTY_PROMPTS:
        prompt += "\n\n" + DIFFICULTY_PROMPTS[difficulty]
    if categories:
        cat_instructions = "\n".join(
            CATEGORY_PROMPTS[c] for c in categories if c in CATEGORY_PROMPTS
        )
        if cat_instructions:
            prompt += "\n\n## Focus areas:\n" + cat_instructions
    if resume_text:
        # Resume is RAG-indexed; instruct the model to retrieve rather than read full text.
        prompt += (
            "\n\n## Candidate Resume (RAG-indexed):\n"
            "The candidate's resume is indexed for semantic search. "
            "Before asking about any skill, technology, or experience area, "
            "call `search_resume` with a relevant query. "
            "Examples: search_resume('Python'), search_resume('system design'), "
            "search_resume('leadership'). "
            "Use results to probe gaps between their background and the job requirements."
        )
    messages.append({"role": "system", "content": prompt})
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if isinstance(content, str) and role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    return messages


def _call_model_streaming(messages: list[dict], client: InferenceClient):
    """Send messages to the model, resolve tool calls, then stream the final response."""
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=MAX_NEW_TOKENS,
    )
    msg = response.choices[0].message

    while msg.tool_calls:
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            messages.append(_execute_tool(tc))
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=MAX_NEW_TOKENS,
        )
        msg = response.choices[0].message

    if msg.content:
        yield msg.content.strip()
        return

    stream = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=MAX_NEW_TOKENS,
        stream=True,
    )
    partial = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial


def _call_model(messages: list[dict], client: InferenceClient) -> str:
    """Non-streaming model call (used for model-answer requests)."""
    result = ""
    for partial in _call_model_streaming(messages, client):
        result = partial
    return result


_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    """Lazy-initialise the HF Inference client."""
    global _client
    if _client is None:
        _client = InferenceClient(token=HF_TOKEN)
    return _client

# ── PDF export ──────────────────────────────────────────────────────────────


def _export_chat_pdf(history: list[dict]) -> str | None:
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
        safe_text = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_text)
        pdf.ln(3)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    return tmp.name

# ── Gradio UI ───────────────────────────────────────────────────────────────


def main() -> None:
    with gr.Blocks(title="Interview Prep Bot") as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(
            f"## Interview Prep Bot &nbsp;·&nbsp; `{MODEL_ID}`\n"
            "Paste a **job description** or URL below to start."
        )

        # ── Settings ────────────────────────────────────────────────────
        with gr.Row():
            difficulty = gr.Radio(
                choices=["Easy", "Medium", "Hard"],
                value="Medium",
                label="Difficulty",
                scale=1,
            )
            categories = gr.CheckboxGroup(
                choices=list(CATEGORY_PROMPTS.keys()),
                value=[],
                label="Focus areas",
                scale=4,
            )
            export_btn = gr.Button("📄 PDF", size="sm", scale=0, min_width=80)

        export_file = gr.File(label="Download transcript", visible=False)

        # ── Documents ───────────────────────────────────────────────────
        with gr.Accordion("📁 Documents", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Job Description**")
                    file_upload = gr.File(
                        label="Upload (PDF, DOCX, or TXT)",
                        file_types=[".pdf", ".docx", ".doc", ".txt"],
                    )
                    upload_btn = gr.Button("Use this file", size="sm")
                with gr.Column():
                    gr.Markdown("**Resume** — enables RAG retrieval")
                    resume_upload = gr.File(
                        label="Upload (PDF, DOCX, or TXT)",
                        file_types=[".pdf", ".docx", ".doc", ".txt"],
                    )
                    resume_btn = gr.Button("Index resume", size="sm")
                    resume_status = gr.Textbox(
                        value="",
                        label="Index status",
                        interactive=False,
                        placeholder="Upload a resume and click 'Index resume'.",
                    )

        # ── Chat ────────────────────────────────────────────────────────
        chatbot = gr.Chatbot(height=540, render_markdown=True)

        # ── Hidden state ────────────────────────────────────────────────
        state = gr.State([])
        question_ts = gr.State(0.0)
        scores = gr.State([])
        resume_state = gr.State("")

        # ── Dynamic input rows (MCQ / rating appear above text input) ───
        with gr.Row(visible=False) as mcq_input_row:
            mcq_radio = gr.Radio(
                choices=[], label="Choose the best answer",
                interactive=True, scale=4,
            )
            mcq_btn = gr.Button("Submit", variant="primary", scale=1)

        with gr.Row(visible=False) as rating_input_row:
            rating_slider = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Your rating", interactive=True, scale=4,
            )
            rating_btn = gr.Button("Submit", variant="primary", scale=1)

        with gr.Row(visible=True) as text_input_row:
            txt = gr.Textbox(
                placeholder="Type your answer, paste a JD, or enter a URL…",
                show_label=False, scale=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # ── Action bar ──────────────────────────────────────────────────
        with gr.Row():
            model_answer_btn = gr.Button("💡 Model Answer", variant="secondary", scale=1)

        # ── Voice input ─────────────────────────────────────────────────
        with gr.Accordion("🎤 Voice Input", open=False):
            audio = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
            voice_btn = gr.Button("Transcribe & Send", size="sm")

        # ── Sessions ────────────────────────────────────────────────────
        with gr.Accordion("💾 Sessions", open=False):
            with gr.Row():
                session_dropdown = gr.Dropdown(
                    choices=[], label="Load past session", scale=3, interactive=True
                )
                load_session_btn = gr.Button("Load", scale=1)
                save_session_btn = gr.Button("Save", scale=1)

        # ── Shared output tuple (must match yield order in handlers) ────
        _outputs = [
            chatbot, state, txt, question_ts, scores,
            text_input_row, mcq_input_row, mcq_radio, rating_input_row, rating_slider,
        ]

        # ── Resume handler ───────────────────────────────────────────────
        def _on_resume_upload(file):
            if file is None:
                return "", ""
            text = _extract_file_text(file)
            if text.startswith("Error"):
                return "", text
            return text, build_resume_index(text)

        resume_btn.click(_on_resume_upload, [resume_upload], [resume_state, resume_status])

        # ── Streaming helper ─────────────────────────────────────────────
        def _stream_reply(user_msg, history, diff, ts, score_list, cats, resume_txt):
            messages = _build_messages(history[:-1], diff, cats, resume_txt)
            messages.append({"role": "user", "content": user_msg})
            partial_history = history + [{"role": "assistant", "content": ""}]

            for partial_text in _call_model_streaming(messages, _get_client()):
                partial_history[-1]["content"] = partial_text
                yield (
                    partial_history, partial_history, "", time.time(), score_list,
                ) + _no_change_visibility()

            final_text = partial_history[-1]["content"]
            clean_text, hint, score = _parse_tags(final_text)
            partial_history[-1]["content"] = clean_text
            if score is not None:
                score_list = score_list + [score]
            yield (
                partial_history, partial_history, "", time.time(), score_list,
            ) + _input_visibility(hint)

        # ── File upload handler ──────────────────────────────────────────
        def _on_file_upload(file, history, diff, ts, score_list, cats, resume_txt):
            if file is None:
                yield (history, history, "", ts, score_list) + _no_change_visibility()
                return
            text = _extract_file_text(file)
            if text.startswith("Error"):
                yield (
                    history + [{"role": "assistant", "content": text}],
                    history + [{"role": "assistant", "content": text}],
                    "", ts, score_list,
                ) + _no_change_visibility()
                return
            user_msg = f"Here is the job description:\n\n{text}"
            history = history + [{"role": "user", "content": user_msg}]
            yield from _stream_reply(user_msg, history, diff, 0.0, score_list, cats, resume_txt)

        upload_btn.click(
            _on_file_upload,
            [file_upload, state, difficulty, question_ts, scores, categories, resume_state],
            _outputs,
        )

        # ── Text submit ──────────────────────────────────────────────────
        def _on_text_submit(message, history, diff, ts, score_list, cats, resume_txt):
            if not message.strip():
                yield (history, history, "", ts, score_list) + _no_change_visibility()
                return
            history = history + [{"role": "user", "content": message}]
            yield from _stream_reply(message, history, diff, ts, score_list, cats, resume_txt)

        txt.submit(
            _on_text_submit,
            [txt, state, difficulty, question_ts, scores, categories, resume_state],
            _outputs,
        )
        send_btn.click(
            _on_text_submit,
            [txt, state, difficulty, question_ts, scores, categories, resume_state],
            _outputs,
        )

        # ── MCQ submit ───────────────────────────────────────────────────
        def _on_mcq_submit(selection, history, diff, ts, score_list, cats, resume_txt):
            if not selection:
                yield (history, history, "", ts, score_list) + _no_change_visibility()
                return
            history = history + [{"role": "user", "content": selection}]
            yield from _stream_reply(selection, history, diff, ts, score_list, cats, resume_txt)

        mcq_btn.click(
            _on_mcq_submit,
            [mcq_radio, state, difficulty, question_ts, scores, categories, resume_state],
            _outputs,
        )

        # ── Rating submit ────────────────────────────────────────────────
        def _on_rating_submit(value, history, diff, ts, score_list, cats, resume_txt):
            if value is None:
                yield (history, history, "", ts, score_list) + _no_change_visibility()
                return
            message = str(int(value))
            history = history + [{"role": "user", "content": message}]
            yield from _stream_reply(message, history, diff, ts, score_list, cats, resume_txt)

        rating_btn.click(
            _on_rating_submit,
            [rating_slider, state, difficulty, question_ts, scores, categories, resume_state],
            _outputs,
        )

        # ── Voice submit ─────────────────────────────────────────────────
        def _on_voice_submit(audio_data, history, diff, ts, score_list, cats, resume_txt):
            text = _transcribe_audio(audio_data)
            if not text:
                yield (history, history, "", ts, score_list) + _no_change_visibility()
                return
            history = history + [{"role": "user", "content": f"[voice] {text}"}]
            yield from _stream_reply(text, history, diff, ts, score_list, cats, resume_txt)

        voice_btn.click(
            _on_voice_submit,
            [audio, state, difficulty, question_ts, scores, categories, resume_state],
            _outputs,
        )

        # ── Model answer ─────────────────────────────────────────────────
        def _on_model_answer(history, diff, cats, resume_txt):
            if not history:
                return history, history
            messages = _build_messages(history, diff, cats, resume_txt)
            messages.append({
                "role": "user",
                "content": (
                    "Give a strong, detailed model answer for the last technical question you asked. "
                    "Format it as if you were the ideal candidate responding. Use code blocks if relevant."
                ),
            })
            reply = _call_model(messages, _get_client())
            history = history + [{"role": "assistant", "content": f"**Model Answer:**\n\n{reply}"}]
            return history, history

        model_answer_btn.click(
            _on_model_answer,
            [state, difficulty, categories, resume_state],
            [chatbot, state],
        )

        # ── PDF export ───────────────────────────────────────────────────
        def _on_export(history):
            path = _export_chat_pdf(history)
            if path is None:
                return gr.update(visible=False)
            return gr.update(value=path, visible=True)

        export_btn.click(_on_export, [state], [export_file])

        # ── Session history (browser localStorage) ───────────────────────
        save_session_js = """
async function(history) {
    if (!history || history.length === 0) return {choices: []};
    const key = 'interview_session_' + Date.now();
    const label = new Date().toLocaleString() + ' (' + history.length + ' msgs)';
    const data = JSON.stringify({label: label, history: history});
    localStorage.setItem(key, data);
    const keys = Object.keys(localStorage).filter(k => k.startsWith('interview_session_'));
    const choices = keys.map(k => {
        try { return JSON.parse(localStorage.getItem(k)).label; } catch(e) { return k; }
    });
    return {choices: choices};
}
"""
        save_session_btn.click(None, [state], [session_dropdown], js=save_session_js)

        load_session_js = """
async function(selected) {
    if (!selected) return [[], []];
    const keys = Object.keys(localStorage).filter(k => k.startsWith('interview_session_'));
    for (const k of keys) {
        try {
            const data = JSON.parse(localStorage.getItem(k));
            if (data.label === selected) return [data.history, data.history];
        } catch(e) {}
    }
    return [[], []];
}
"""
        load_session_btn.click(None, [session_dropdown], [chatbot, state], js=load_session_js)

        populate_sessions_js = """
async function() {
    const keys = Object.keys(localStorage).filter(k => k.startsWith('interview_session_'));
    const choices = keys.map(k => {
        try { return JSON.parse(localStorage.getItem(k)).label; } catch(e) { return k; }
    });
    return {choices: choices.length > 0 ? choices : []};
}
"""
        demo.load(None, [], [session_dropdown], js=populate_sessions_js)

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
