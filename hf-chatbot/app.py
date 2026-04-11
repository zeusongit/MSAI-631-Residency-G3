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

import os
import time

import gradio as gr

from chat import build_messages, call_model, call_model_streaming, get_client
from config import CATEGORY_PROMPTS, MODEL_ID
from export import export_chat_pdf
from file_parsing import extract_file_text
from scoring import fmt_avg, format_duration, parse_score
from whisper import transcribe_audio

# ── Gradio UI ───────────────────────────────────────────────────────────────


def main() -> None:
    with gr.Blocks(title="Interview Prep Bot") as demo:
        gr.Markdown(
            "# Interview Prep Bot\n"
            f"Model: `{MODEL_ID}` via Inference API &nbsp;|&nbsp; "
            "Voice input powered by Whisper\n\n"
            "Paste a **job description**, upload a file, or provide a URL to start."
        )

        # ── Session history ────────────────────────────────────────────

        with gr.Row():
            session_dropdown = gr.Dropdown(
                choices=[], label="Load past session", scale=3, interactive=True
            )
            load_session_btn = gr.Button("Load", scale=1)
            save_session_btn = gr.Button("Save session", scale=1)

        # ── Settings row ────────────────────────────────────────────────

        with gr.Row():
            difficulty = gr.Radio(
                choices=["Easy", "Medium", "Hard"],
                value="Medium",
                label="Difficulty",
                scale=2,
            )
            categories = gr.CheckboxGroup(
                choices=list(CATEGORY_PROMPTS.keys()),
                value=[],
                label="Focus areas (optional)",
                scale=3,
            )

        with gr.Row():
            timer_display = gr.Textbox(
                value="",
                label="Answer time",
                interactive=False,
                scale=1,
            )
            score_display = gr.Textbox(
                value="",
                label="Avg score",
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

        with gr.Accordion("Upload your resume (optional)", open=False):
            resume_upload = gr.File(
                label="Upload resume (PDF, DOCX, or TXT)",
                file_types=[".pdf", ".docx", ".doc", ".txt"],
            )
            resume_btn = gr.Button("Use this resume")

        # ── Chat area ──────────────────────────────────────────────────

        chatbot = gr.Chatbot(height=450, render_markdown=True)
        state = gr.State([])  # chat history as list[dict]
        question_ts = gr.State(0.0)  # timestamp when last bot message was shown
        scores = gr.State([])  # list of int scores
        resume_state = gr.State("")  # resume text

        with gr.Tabs() as input_tabs:
            with gr.Tab("Text", id="text_tab"):
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Type your answer (or paste a JD / URL to begin)...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Tab("Code Editor", id="code_tab"):
                with gr.Row():
                    code_lang = gr.Dropdown(
                        choices=["python", "javascript", "typescript", "java", "c++", "c", "go", "rust", "sql", "bash", "html", "css", "other"],
                        value="python",
                        label="Language",
                        scale=1,
                    )
                code_input = gr.Code(
                    language="python",
                    label="Write your code here",
                    lines=12,
                )
                with gr.Row():
                    code_explanation = gr.Textbox(
                        placeholder="(Optional) Explain your approach...",
                        show_label=False,
                        scale=4,
                    )
                    send_code_btn = gr.Button("Send Code", variant="primary", scale=1)

        with gr.Row():
            model_answer_btn = gr.Button("Show model answer", variant="secondary")

        # ── Voice input ─────────────────────────────────────────────────

        with gr.Accordion("Voice input", open=False):
            audio = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
            voice_btn = gr.Button("Transcribe & Send")

        # ── Resume upload handler ──────────────────────────────────────

        def _on_resume_upload(file):
            if file is None:
                return ""
            text = extract_file_text(file)
            if text.startswith("Error"):
                return ""
            return text

        resume_btn.click(
            _on_resume_upload,
            [resume_upload],
            [resume_state],
        )

        # ── File upload handler ─────────────────────────────────────────

        def _on_file_upload(file, history: list[dict], diff: str,
                            score_list: list, cats: list[str], resume_txt: str):
            if file is None:
                yield history, history, "", 0.0, score_list, fmt_avg(score_list)
                return
            text = extract_file_text(file)
            if text.startswith("Error"):
                history = history + [{"role": "assistant", "content": text}]
                yield history, history, "", 0.0, score_list, fmt_avg(score_list)
                return
            user_msg = f"Here is the job description:\n\n{text}"
            history = history + [{"role": "user", "content": user_msg}]
            messages = build_messages(history[:-1], diff, cats, resume_txt)
            messages.append({"role": "user", "content": user_msg})

            partial_history = history + [{"role": "assistant", "content": ""}]
            for partial_text in call_model_streaming(messages, get_client()):
                partial_history[-1]["content"] = partial_text
                yield (partial_history, partial_history, "", time.time(),
                       score_list, fmt_avg(score_list))

            final_text = partial_history[-1]["content"]
            clean_text, score = parse_score(final_text)
            partial_history[-1]["content"] = clean_text
            if score is not None:
                score_list = score_list + [score]
            yield (partial_history, partial_history, "", time.time(),
                   score_list, fmt_avg(score_list))

        upload_btn.click(
            _on_file_upload,
            [file_upload, state, difficulty, scores, categories, resume_state],
            [chatbot, state, timer_display, question_ts, scores, score_display],
        )

        # ── Text submit ─────────────────────────────────────────────────

        def _on_text_submit(message: str, history: list[dict], diff: str, ts: float,
                            score_list: list, cats: list[str], resume_txt: str):
            if not message.strip():
                yield history, history, "", "", ts, score_list, fmt_avg(score_list)
                return
            duration_str = ""
            if ts > 0 and history and history[-1].get("role") == "assistant":
                elapsed = time.time() - ts
                duration_str = format_duration(elapsed)

            history = history + [{"role": "user", "content": message}]
            messages = build_messages(history[:-1], diff, cats, resume_txt)
            messages.append({"role": "user", "content": message})

            partial_history = history + [{"role": "assistant", "content": ""}]
            for partial_text in call_model_streaming(messages, get_client()):
                partial_history[-1]["content"] = partial_text
                yield (partial_history, partial_history, "", duration_str, time.time(),
                       score_list, fmt_avg(score_list))

            final_text = partial_history[-1]["content"]
            clean_text, score = parse_score(final_text)
            partial_history[-1]["content"] = clean_text
            if score is not None:
                score_list = score_list + [score]
            yield (partial_history, partial_history, "", duration_str, time.time(),
                   score_list, fmt_avg(score_list))

        txt.submit(
            _on_text_submit,
            [txt, state, difficulty, question_ts, scores, categories, resume_state],
            [chatbot, state, txt, timer_display, question_ts, scores, score_display],
        )
        send_btn.click(
            _on_text_submit,
            [txt, state, difficulty, question_ts, scores, categories, resume_state],
            [chatbot, state, txt, timer_display, question_ts, scores, score_display],
        )

        # ── Code submit ────────────────────────────────────────────────

        def _on_code_submit(code: str, explanation: str, lang: str,
                            history: list[dict], diff: str, ts: float,
                            score_list: list, cats: list[str], resume_txt: str):
            if not code or not code.strip():
                yield history, history, None, "", "", ts, score_list, fmt_avg(score_list)
                return
            duration_str = ""
            if ts > 0 and history and history[-1].get("role") == "assistant":
                elapsed = time.time() - ts
                duration_str = format_duration(elapsed)

            # Format the code as a markdown code block with optional explanation
            message = f"```{lang}\n{code}\n```"
            if explanation and explanation.strip():
                message = f"{explanation.strip()}\n\n{message}"

            history = history + [{"role": "user", "content": message}]
            messages = build_messages(history[:-1], diff, cats, resume_txt)
            messages.append({"role": "user", "content": message})

            partial_history = history + [{"role": "assistant", "content": ""}]
            for partial_text in call_model_streaming(messages, get_client()):
                partial_history[-1]["content"] = partial_text
                yield (partial_history, partial_history, None, "", duration_str, time.time(),
                       score_list, fmt_avg(score_list))

            final_text = partial_history[-1]["content"]
            clean_text, score = parse_score(final_text)
            partial_history[-1]["content"] = clean_text
            if score is not None:
                score_list = score_list + [score]
            yield (partial_history, partial_history, None, "", duration_str, time.time(),
                   score_list, fmt_avg(score_list))

        send_code_btn.click(
            _on_code_submit,
            [code_input, code_explanation, code_lang, state, difficulty,
             question_ts, scores, categories, resume_state],
            [chatbot, state, code_input, code_explanation, timer_display,
             question_ts, scores, score_display],
        )

        # Update code editor language when dropdown changes
        def _update_code_lang(lang):
            return gr.Code(language=lang if lang != "other" else None)

        code_lang.change(_update_code_lang, [code_lang], [code_input])

        # ── Voice submit ────────────────────────────────────────────────

        def _on_voice_submit(audio_data, history: list[dict], diff: str, ts: float,
                             score_list: list, cats: list[str], resume_txt: str):
            text = transcribe_audio(audio_data)
            if not text:
                yield history, history, None, "", ts, score_list, fmt_avg(score_list)
                return
            duration_str = ""
            if ts > 0 and history and history[-1].get("role") == "assistant":
                elapsed = time.time() - ts
                duration_str = format_duration(elapsed)

            history = history + [{"role": "user", "content": f"[voice] {text}"}]
            messages = build_messages(history[:-1], diff, cats, resume_txt)
            messages.append({"role": "user", "content": text})

            partial_history = history + [{"role": "assistant", "content": ""}]
            for partial_text in call_model_streaming(messages, get_client()):
                partial_history[-1]["content"] = partial_text
                yield (partial_history, partial_history, None, duration_str, time.time(),
                       score_list, fmt_avg(score_list))

            final_text = partial_history[-1]["content"]
            clean_text, score = parse_score(final_text)
            partial_history[-1]["content"] = clean_text
            if score is not None:
                score_list = score_list + [score]
            yield (partial_history, partial_history, None, duration_str, time.time(),
                   score_list, fmt_avg(score_list))

        voice_btn.click(
            _on_voice_submit,
            [audio, state, difficulty, question_ts, scores, categories, resume_state],
            [chatbot, state, audio, timer_display, question_ts, scores, score_display],
        )

        # ── Model answer ───────────────────────────────────────────────

        def _on_model_answer(history: list[dict], diff: str, cats: list[str], resume_txt: str):
            if not history:
                return history, history
            messages = build_messages(history, diff, cats, resume_txt)
            messages.append({
                "role": "user",
                "content": "Give a strong, detailed model answer for the last technical question you asked. "
                           "Format it as if you were the ideal candidate responding. Use code blocks if relevant.",
            })
            reply = call_model(messages, get_client())
            reply = f"**Model Answer:**\n\n{reply}"
            history = history + [{"role": "assistant", "content": reply}]
            return history, history

        model_answer_btn.click(
            _on_model_answer,
            [state, difficulty, categories, resume_state],
            [chatbot, state],
        )

        # ── PDF export ──────────────────────────────────────────────────

        def _on_export(history: list[dict]):
            path = export_chat_pdf(history)
            if path is None:
                return gr.update(visible=False)
            return gr.update(value=path, visible=True)

        export_btn.click(_on_export, [state], [export_file])

        # ── Session history (browser localStorage) ─────────────────────

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

        save_session_btn.click(
            None, [state], [session_dropdown], js=save_session_js
        )

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

        load_session_btn.click(
            None, [session_dropdown], [chatbot, state], js=load_session_js
        )

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
