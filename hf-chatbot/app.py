"""
Interview Prep Chatbot — Gradio UI powered by Hugging Face Inference API.

Given a job description (pasted text, URL, or uploaded file), the bot asks
tailored technical and behavioral interview questions, evaluates answers
with constructive feedback, and asks follow-up questions before moving on.

Supports voice input via browser microphone (transcribed with Whisper),
file upload (PDF/DOCX/TXT), difficulty selection, answer timing, and
session export as PDF.

Set HUGGING_FACE_HUB_TOKEN for API access. See README for all env vars.
"""

from __future__ import annotations

import io
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
        "and simple scenario-based behavioral questions. Be extra supportive."
    ),
    "Medium": (
        "Adjust your questions to a MID-LEVEL difficulty. "
        "Ask questions that require solid understanding, some depth, "
        "and real-world examples. Balance challenge with encouragement."
    ),
    "Hard": (
        "Adjust your questions to a SENIOR / STAFF-LEVEL difficulty. "
        "Ask deep technical questions, complex system design scenarios, "
        "and behavioral questions that probe leadership and conflict resolution. "
        "Be rigorous — this person wants to be pushed."
    ),
}

SYSTEM_PROMPT = os.environ.get("HF_SYSTEM_PROMPT", """\
You are a friendly, experienced interview coach having a natural conversation. \
You speak like a real person — warm, supportive, and direct. Never dump lists \
of questions or walls of text.

## Your personality:
- Conversational and natural — like chatting with a knowledgeable friend
- Keep responses short (2-4 sentences typical, never more than a short paragraph)
- Use casual but professional language
- React to what the user actually said before moving on
- Show genuine interest in their answers

## Flow:

**Getting started:** When the user shares a job description (text or URL — use \
fetch_url for URLs), read it carefully. Then give a brief, friendly take on the \
role (2-3 sentences max). Use web_search to quietly research the company so your \
questions feel informed. Ask if they're ready to start.

**During the interview:** Ask ONE question at a time. After they answer:
- Acknowledge what they said specifically ("That's a solid example of..." or \
"I like that you mentioned...")
- Give one concrete tip to strengthen the answer if needed
- Then naturally transition to a follow-up or the next topic

Do NOT number your questions. Do NOT say "Question 1:" or "Let's move to question 2." \
Just flow naturally, like a real conversation. Sometimes the follow-up should come \
from something interesting they said, not from a pre-planned list.

Mix technical and behavioral questions, but weave them in naturally. A good interviewer \
doesn't announce "now for a behavioral question" — they just ask it.

**Wrapping up:** After 5-7 questions (don't count out loud), naturally wind down: \
"I think that's a good place to wrap up." Then give a brief, honest summary — \
what they're strong on, what to work on, and one or two specific tips for this \
particular role. Keep it encouraging but real. Offer to save the summary.

## Important:
- NEVER list multiple questions at once
- NEVER use numbered steps or bullet-point feedback dumps
- NEVER say things like "Let me evaluate your answer" — just react naturally
- If an answer is weak, coach them through it: "What if you also mentioned..."
- If they say "skip" or "next", just smoothly move on
- If they say "done" or "end", wrap up with the summary
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
    # Normalise to float32 mono
    audio = audio_data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.max() > 1.0:
        audio = audio / 32768.0
    # Write to a temp WAV for faster-whisper
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

# ── Chat logic ──────────────────────────────────────────────────────────────


def _build_messages(history: list[dict], difficulty: str) -> list[dict]:
    """Assemble the full message list from Gradio chat history."""
    messages: list[dict] = []
    prompt = SYSTEM_PROMPT.strip()
    if difficulty in DIFFICULTY_PROMPTS:
        prompt += "\n\n" + DIFFICULTY_PROMPTS[difficulty]
    messages.append({"role": "system", "content": prompt})
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if isinstance(content, str) and role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    return messages


def _call_model(messages: list[dict], client: InferenceClient) -> str:
    """Send messages to the model, execute any tool calls, return final text."""
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=MAX_NEW_TOKENS,
    )
    msg = response.choices[0].message

    # Resolve tool calls iteratively
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

    return (msg.content or "").strip()


_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    """Lazy-initialise the HF Inference client."""
    global _client
    if _client is None:
        _client = InferenceClient(token=HF_TOKEN)
    return _client


def _respond(message: str, history: list[dict], difficulty: str) -> str:
    """Generate the next assistant reply."""
    messages = _build_messages(history, difficulty)
    messages.append({"role": "user", "content": message})
    return _call_model(messages, _get_client())

# ── Answer timer helpers ────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    """Format seconds into a readable string like '1m 23s'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

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
        # Encode to latin-1 for fpdf, replacing unsupported chars
        safe_text = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_text)
        pdf.ln(3)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    return tmp.name

# ── Gradio UI ───────────────────────────────────────────────────────────────


def main() -> None:
    with gr.Blocks(title="Interview Prep Bot") as demo:
        gr.Markdown(
            "# Interview Prep Bot\n"
            f"Model: `{MODEL_ID}` via Inference API &nbsp;|&nbsp; "
            "Voice input powered by Whisper\n\n"
            "Paste a **job description**, upload a file, or provide a URL to start."
        )

        # ── Settings row ────────────────────────────────────────────────

        with gr.Row():
            difficulty = gr.Radio(
                choices=["Easy", "Medium", "Hard"],
                value="Medium",
                label="Difficulty",
                scale=2,
            )
            timer_display = gr.Textbox(
                value="",
                label="Answer time",
                interactive=False,
                scale=1,
            )
            export_btn = gr.Button("Export as PDF", scale=1)
            export_file = gr.File(label="Download", visible=False)

        # ── File upload ─────────────────────────────────────────────────

        with gr.Accordion("Upload a job description file", open=False):
            file_upload = gr.File(
                label="Upload JD (PDF, DOCX, or TXT)",
                file_types=[".pdf", ".docx", ".doc", ".txt"],
            )
            upload_btn = gr.Button("Use this file")

        # ── Chat area ──────────────────────────────────────────────────

        chatbot = gr.Chatbot(height=450)
        state = gr.State([])  # chat history as list[dict]
        question_ts = gr.State(0.0)  # timestamp when last bot message was shown

        with gr.Row():
            txt = gr.Textbox(
                placeholder="Type your answer (or paste a JD / URL to begin)...",
                show_label=False,
                scale=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # ── Voice input ─────────────────────────────────────────────────

        with gr.Accordion("Voice input", open=False):
            audio = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
            voice_btn = gr.Button("Transcribe & Send")

        # ── File upload handler ─────────────────────────────────────────

        def _on_file_upload(file, history: list[dict], diff: str):
            if file is None:
                return history, history, "", 0.0
            text = _extract_file_text(file)
            if text.startswith("Error"):
                history = history + [{"role": "assistant", "content": text}]
                return history, history, "", 0.0
            user_msg = f"Here is the job description:\n\n{text}"
            history = history + [{"role": "user", "content": user_msg}]
            reply = _respond(user_msg, history[:-1], diff)
            history = history + [{"role": "assistant", "content": reply}]
            return history, history, "", time.time()

        upload_btn.click(
            _on_file_upload,
            [file_upload, state, difficulty],
            [chatbot, state, timer_display, question_ts],
        )

        # ── Text submit ─────────────────────────────────────────────────

        def _on_text_submit(message: str, history: list[dict], diff: str, ts: float):
            if not message.strip():
                return history, history, "", "", ts
            # Calculate answer time if there's a previous bot question
            duration_str = ""
            if ts > 0 and history and history[-1].get("role") == "assistant":
                elapsed = time.time() - ts
                duration_str = _format_duration(elapsed)

            history = history + [{"role": "user", "content": message}]
            reply = _respond(message, history[:-1], diff)
            history = history + [{"role": "assistant", "content": reply}]
            return history, history, "", duration_str, time.time()

        txt.submit(
            _on_text_submit,
            [txt, state, difficulty, question_ts],
            [chatbot, state, txt, timer_display, question_ts],
        )
        send_btn.click(
            _on_text_submit,
            [txt, state, difficulty, question_ts],
            [chatbot, state, txt, timer_display, question_ts],
        )

        # ── Voice submit ────────────────────────────────────────────────

        def _on_voice_submit(audio_data, history: list[dict], diff: str, ts: float):
            text = _transcribe_audio(audio_data)
            if not text:
                return history, history, None, "", ts
            duration_str = ""
            if ts > 0 and history and history[-1].get("role") == "assistant":
                elapsed = time.time() - ts
                duration_str = _format_duration(elapsed)

            history = history + [{"role": "user", "content": f"[voice] {text}"}]
            reply = _respond(text, history[:-1], diff)
            history = history + [{"role": "assistant", "content": reply}]
            return history, history, None, duration_str, time.time()

        voice_btn.click(
            _on_voice_submit,
            [audio, state, difficulty, question_ts],
            [chatbot, state, audio, timer_display, question_ts],
        )

        # ── PDF export ──────────────────────────────────────────────────

        def _on_export(history: list[dict]):
            path = _export_chat_pdf(history)
            if path is None:
                return gr.update(visible=False)
            return gr.update(value=path, visible=True)

        export_btn.click(_on_export, [state], [export_file])

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
