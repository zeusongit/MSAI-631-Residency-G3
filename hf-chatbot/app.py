"""
Interview Prep Chatbot — Gradio UI powered by Hugging Face Inference API.

Given a job description (pasted text, URL, or file upload), the bot asks
tailored technical and behavioral interview questions, evaluates answers
with constructive feedback, and asks follow-up questions before moving on.

Supports voice input via browser microphone (transcribed with Whisper).

Set HUGGING_FACE_HUB_TOKEN for API access. See README for all env vars.
"""

from __future__ import annotations

import json
import os
import re
import tempfile

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


def _build_messages(history: list[dict]) -> list[dict]:
    """Assemble the full message list from Gradio chat history."""
    messages: list[dict] = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
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


def respond(message: str, history: list[dict]) -> str:
    """Gradio ChatInterface callback — generate the next assistant reply."""
    messages = _build_messages(history)
    messages.append({"role": "user", "content": message})
    return _call_model(messages, _get_client())


def transcribe_and_respond(audio, history: list[dict]) -> tuple[str, str]:
    """Transcribe voice input, then generate a response."""
    text = _transcribe_audio(audio)
    if not text:
        return "", "Could not transcribe audio. Please try again or type your answer."
    reply = respond(text, history)
    return text, reply

# ── Gradio UI ───────────────────────────────────────────────────────────────


def main() -> None:
    with gr.Blocks(title="Interview Prep Bot") as demo:
        gr.Markdown(
            "# Interview Prep Bot\n"
            f"Model: `{MODEL_ID}` via Inference API &nbsp;|&nbsp; "
            "Voice input powered by Whisper\n\n"
            "Paste a **job description** (or a URL to one) to start your mock interview."
        )

        chatbot = gr.Chatbot(height=500)
        state = gr.State([])  # chat history as list[dict]

        with gr.Row():
            txt = gr.Textbox(
                placeholder="Type your answer (or paste a JD / URL to begin)...",
                show_label=False,
                scale=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Voice input", open=False):
            audio = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
            voice_btn = gr.Button("Transcribe & Send")

        # ── Text submit ─────────────────────────────────────────────────

        def _on_text_submit(message: str, history: list[dict]):
            if not message.strip():
                return history, history, ""
            history = history + [{"role": "user", "content": message}]
            reply = respond(message, history[:-1])  # history before this msg
            history = history + [{"role": "assistant", "content": reply}]
            return history, history, ""

        txt.submit(_on_text_submit, [txt, state], [chatbot, state, txt])
        send_btn.click(_on_text_submit, [txt, state], [chatbot, state, txt])

        # ── Voice submit ────────────────────────────────────────────────

        def _on_voice_submit(audio_data, history: list[dict]):
            text = _transcribe_audio(audio_data)
            if not text:
                return history, history, None
            history = history + [{"role": "user", "content": f"[voice] {text}"}]
            reply = respond(text, history[:-1])
            history = history + [{"role": "assistant", "content": reply}]
            return history, history, None

        voice_btn.click(_on_voice_submit, [audio, state], [chatbot, state, audio])

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
