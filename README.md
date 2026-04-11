# Interview Prep Chatbot

**Gradio + Hugging Face** interview coaching bot. Paste a job description (or URL) and the bot conducts a tailored mock interview with technical and behavioral questions, evaluates your answers, asks follow-ups, and provides a session summary.

**Features:**
- Tool calling — fetches JD from URLs, searches the web for company context, saves session summaries
- Voice input — record answers via browser microphone, transcribed locally with Whisper
- Adaptive follow-ups — probes deeper before moving to the next topic

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
