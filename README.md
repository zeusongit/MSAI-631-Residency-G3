# HF chatbot — initial setup

Small **Gradio + Transformers** chat UI using [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (CPU-friendly, Apache-2.0).

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

3. **Install dependencies** (from this `hf-chatbot` directory):

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Hugging Face token** — only needed for **gated** models or higher rate limits. Do not commit it.

   ```bash
   export HUGGING_FACE_HUB_TOKEN="hf_..." # macOS/Linux
   ```

5. **Run the app:**

   ```bash
   python app.py
   ```

   Open the URL Gradio prints (usually `http://127.0.0.1:7860`).

6. **Optional environment variables**

   | Variable | Purpose |
   |----------|---------|
   | `HF_MODEL_ID` | Override model (default: SmolLM2-360M-Instruct) |
   | `HF_SYSTEM_PROMPT` | System / behavior instructions |
   | `HF_MAX_NEW_TOKENS` | Cap reply length (default: 256) |
   | `GRADIO_SERVER_PORT` | Port (default: 7860) |

## Hugging Face Space (host the same app)

1. Log in at [huggingface.co](https://huggingface.co) and create a **New Space** (SDK: **Gradio**).
2. Upload **`app.py`** and **`requirements.txt`** from this folder to the Space repo (web UI “Add file”, or Git clone/push).
3. **Hardware:** start with **CPU**; upgrade to free GPU only if the Space supports it and your model needs it.
4. For gated models, add a **Space secret** (Settings → Secrets) named `HUGGING_FACE_HUB_TOKEN` with your token.

## Git

Use GitHub, the Space repo, or both; keep one **canonical** remote and document it for your team.
