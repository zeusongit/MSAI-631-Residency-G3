# Interview Prep Chatbot

**Gradio + Hugging Face** technical interview coaching bot. Paste a job description (or URL), optionally upload your resume, and the bot conducts a tailored mock interview focused on technical depth. It scores your answers, streams responses in real time, and lets you save sessions for later review.

## Project Structure

```
MSAI-631-Residency-G3/
├── README.md
├── LICENSE
└── hf-chatbot/
    ├── app.py               # Gradio UI, event handlers, main entry point
    ├── config.py            # Constants, prompts, environment variables
    ├── chat.py              # LLM client, message building, streaming
    ├── tools.py             # Tool definitions, URL scraping, web search
    ├── scoring.py           # Answer score parsing, averages, timer
    ├── whisper.py           # Voice transcription (faster-whisper)
    ├── file_parsing.py      # PDF/DOCX/TXT text extraction
    ├── export.py            # PDF session export
    ├── requirements.txt     # Python dependencies
    └── .gitignore
```

## Architecture

Modular Gradio app split across logical files. The LLM runs remotely via the HF Inference API; only Whisper runs locally for voice transcription. No backend database or separate server.

```
Browser (Gradio UI)
  ├─ Text / Code Editor / Voice / File input
  ├─ localStorage (session save/load)
  └─ Markdown rendering (code blocks)
        │
        ▼
   app.py ─── Gradio event handlers
  ├─ chat.py
  │   ├─ build_messages()          → assembles system prompt + difficulty + categories + resume + history
  │   └─ call_model_streaming()    → streams tokens from HF Inference API, resolves tool calls
  ├─ tools.py
  │   ├─ fetch_url()               → multi-strategy URL scraping (requests → Playwright → search)
  │   └─ web_search()              → DuckDuckGo search
  ├─ scoring.py
  │   └─ parse_score()             → extracts hidden <!--SCORE:N--> tags from responses
  ├─ whisper.py                    → local speech-to-text
  └─ config.py                     → all prompts and settings
        │
        ▼
   HF Inference API (Qwen/Qwen2.5-Coder-32B-Instruct)
```

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| **Gradio** | Web UI framework with streaming chatbot, file upload, audio input |
| **HuggingFace Inference API** | Remote LLM inference (Qwen2.5-Coder-32B-Instruct) |
| **faster-whisper** | Local speech-to-text for voice input |
| **Playwright** | Headless Chrome browser for scraping JS-rendered job sites |
| **BeautifulSoup** | HTML parsing, JSON-LD extraction, CSS selector matching |
| **ddgs** | DuckDuckGo web search for company/role research |
| **fpdf2** | PDF export of interview sessions |
| **pypdf / python-docx** | Parse uploaded JD and resume files (PDF/DOCX) |

## URL Scraping

When you paste a job posting URL, the bot extracts the description using a multi-strategy pipeline:

1. **requests** — fast HTTP fetch with browser-like headers
2. **Playwright** — if the site returns 403/429, spins up headless Chrome to render the page with JavaScript
3. **JSON-LD** — extracts structured `JobPosting` data embedded in the page (Indeed, LinkedIn, many job boards)
4. **CSS selectors** — targets common job description containers (`#jobDescriptionText`, `.job-description`, etc.)
5. **Full page text** — strips nav/script/style and returns the body text
6. **DuckDuckGo search** — last resort if all above fail, searches for the job title + location

| Site | Method | Status |
|------|--------|--------|
| **Indeed** | Playwright + JSON-LD | Works |
| **Greenhouse** | Direct HTML | Works |
| **Lever** | Direct HTML | Works |
| **Company career pages** | Direct HTML | Works |
| **LinkedIn** | Requires login | Fallback to search |
| **Glassdoor** | Playwright | May work, depends on region |

## Features

- **Technical-only questions** — no small talk, jumps straight into role-relevant technical questions
- **Resume upload** — tailors questions to gaps between your experience and the JD
- **Category focus areas** — multi-select: System Design, Coding/Algorithms, Domain Knowledge, API Design, Debugging, Architecture
- **Answer scoring** — hidden 1-5 scoring per answer with running average display
- **Code editor** — syntax-highlighted editor with language selector for coding questions
- **Model answer** — request an ideal example answer for any question
- **Streaming responses** — real-time token streaming for all input methods
- **Voice input** — browser microphone, transcribed locally with Whisper
- **Session history** — save/load past sessions via browser localStorage
- **Code block support** — markdown rendering with syntax highlighting
- **Tool calling** — fetches JD from URLs, searches the web for company context
- **PDF export** — download your full interview session
- **Difficulty levels** — Easy / Medium / Hard

**Model:** [`Qwen/Qwen2.5-Coder-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) via HF Inference API (override with `HF_MODEL_ID`).

## Setup

### Prerequisites

- Python 3.10+
- A [Hugging Face API token](https://huggingface.co/settings/tokens) with **"Make calls to the serverless Inference API"** enabled

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/zeusongit/MSAI-631-Residency-G3.git
cd MSAI-631-Residency-G3/hf-chatbot

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate.bat     # Windows CMD
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install headless Chrome for job site scraping
playwright install chromium

# 5. Set your Hugging Face API token
export HUGGING_FACE_HUB_TOKEN="hf_..."   # macOS/Linux
# set HUGGING_FACE_HUB_TOKEN=hf_...      # Windows CMD

# 6. Run the app
python app.py
```

Open the URL Gradio prints (usually `http://127.0.0.1:7860`).

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HUGGING_FACE_HUB_TOKEN` | HF API token | (required) |
| `HF_MODEL_ID` | Override chat model | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `HF_SYSTEM_PROMPT` | Override the interview coach system prompt | (built-in) |
| `HF_MAX_NEW_TOKENS` | Cap reply length | `1024` |
| `HF_WHISPER_MODEL` | Whisper model size for voice input | `small` |
| `GRADIO_SERVER_PORT` | Port | `7860` |

## Usage

1. Open the web UI in your browser.
2. Paste a job description, URL, or upload a JD file and hit Send.
3. Optionally upload your resume and select focus areas / difficulty.
4. Answer questions using the **Text** tab or **Code Editor** tab (for coding questions).
5. Use **Voice input** to record answers via your microphone.
6. Click **Show model answer** to see an ideal response for any question.
7. Say "done" to get a session summary with strengths and areas to improve.
8. **Save session** to browser localStorage, or **Export as PDF**.

## Hugging Face Space

To host the app on Hugging Face Spaces:

1. Create a **New Space** (SDK: **Gradio**) at [huggingface.co](https://huggingface.co).
2. Upload all `.py` files and `requirements.txt` from the `hf-chatbot` folder.
3. **Hardware:** start with **CPU** — inference happens via the API, not locally (only Whisper runs locally).
4. Add a **Space secret** named `HUGGING_FACE_HUB_TOKEN` with your token.
