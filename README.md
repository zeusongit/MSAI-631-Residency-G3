# Interview Prep Chatbot

**Gradio + Hugging Face** technical interview coaching bot. Paste a job description (or URL), optionally upload your resume, and the bot conducts a tailored mock interview focused on technical depth. It scores your answers, streams responses in real time, and lets you save sessions for later review.

## Architecture

Single-file Gradio app (`hf-chatbot/app.py`) — no backend database or separate server. The LLM runs remotely via the HF Inference API; only Whisper runs locally for voice transcription.

```
Browser (Gradio UI)
  ├─ Text / Voice / File input
  ├─ localStorage (session save/load)
  └─ Markdown rendering (code blocks)
        │
        ▼
   app.py (Python, aiohttp)
  ├─ _build_messages()  → assembles system prompt + difficulty + categories + resume + history
  ├─ _call_model_streaming() → streams tokens from HF Inference API, resolves tool calls
  ├─ _parse_score()     → extracts hidden <!--SCORE:N--> tags from responses
  ├─ Tools: fetch_url, web_search, save_session
  └─ Whisper (faster-whisper, local)
        │
        ▼
   HF Inference API (Qwen/Qwen2.5-72B-Instruct)
```

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| **Gradio** | Web UI framework with streaming chatbot, file upload, audio input |
| **HuggingFace Inference API** | Remote LLM inference (Qwen2.5-72B-Instruct) |
| **faster-whisper** | Local speech-to-text for voice input |
| **DuckDuckGo Search** | Web search tool for company/role research |
| **fpdf2** | PDF export of interview sessions |
| **pypdf / python-docx** | Parse uploaded JD and resume files (PDF/DOCX) |

## Features

- **Technical-only questions** — no small talk, jumps straight into role-relevant technical questions
- **Resume upload** — tailors questions to gaps between your experience and the JD
- **Category focus areas** — multi-select: System Design, Coding/Algorithms, Domain Knowledge, API Design, Debugging, Architecture
- **Answer scoring** — hidden 1-5 scoring per answer with running average display
- **Model answer** — request an ideal example answer for any question
- **Streaming responses** — real-time token streaming for all input methods
- **Voice input** — browser microphone, transcribed locally with Whisper
- **Session history** — save/load past sessions via browser localStorage
- **Code block support** — markdown rendering with syntax highlighting
- **Tool calling** — fetches JD from URLs, searches the web for company context
- **PDF export** — download your full interview session
- **Difficulty levels** — Easy / Medium / Hard

**Model:** [`Qwen/Qwen2.5-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) via HF Inference API (override with `HF_MODEL_ID`).

## Dev environment (local)

1. **Python 3.10+** installed.

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

   Activate it:

   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows (Command Prompt):**
     ```bat
     .venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

   > If `python -m venv .venv` fails due to a colon (`:`) in your path, create the venv outside the project:
   > ```bash
   > python3 -m venv ~/venvs/hf-chatbot && source ~/venvs/hf-chatbot/bin/activate
   > ```
   > then `cd` back into this folder.

3. **Install dependencies** (from the `hf-chatbot` directory):

   ```bash
   pip install -r requirements.txt
   ```

4. **Hugging Face token** — required for the Inference API. Do not commit it.

   ```bash
   export HUGGING_FACE_HUB_TOKEN="hf_..."   # macOS/Linux
   ```

   Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Enable the **"Make calls to the serverless Inference API"** permission.

5. **Run the app:**

   ```bash
   python app.py
   ```

   Open the URL Gradio prints (usually `http://127.0.0.1:7860`).

6. **Environment variables**

   | Variable | Purpose |
   |----------|---------|
   | `HUGGING_FACE_HUB_TOKEN` | HF API token (required) |
   | `HF_MODEL_ID` | Override chat model (default: Qwen/Qwen2.5-72B-Instruct) |
   | `HF_SYSTEM_PROMPT` | Override the interview coach system prompt |
   | `HF_MAX_NEW_TOKENS` | Cap reply length (default: 1024) |
   | `HF_WHISPER_MODEL` | Whisper model size for voice input (default: small) |
   | `GRADIO_SERVER_PORT` | Port (default: 7860) |

## Usage

1. Open the web UI in your browser.
2. Paste a job description or a URL to a job posting in the text box and hit Send.
3. The bot analyzes the JD, researches the company, and starts asking interview questions.
4. Answer by typing or click **Voice input** to record via your microphone.
5. After each answer you get feedback and follow-up questions.
6. Say "done" to get a session summary with strengths and areas to improve.

## Hugging Face Space (host the same app)

1. Log in at [huggingface.co](https://huggingface.co) and create a **New Space** (SDK: **Gradio**).
2. Upload **`app.py`** and **`requirements.txt`** from the `hf-chatbot` folder to the Space repo.
3. **Hardware:** start with **CPU** — inference happens via the API, not locally (only Whisper runs locally).
4. Add a **Space secret** named `HUGGING_FACE_HUB_TOKEN` with your token.

## Git

Use GitHub, the Space repo, or both; keep one **canonical** remote and document it for your team.
