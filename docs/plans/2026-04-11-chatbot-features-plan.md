# Interview Prep Bot: 7 Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add resume upload, question categories, answer scoring, model answer button, session history (localStorage), streaming responses, and code block support to the interview prep chatbot.

**Architecture:** All changes in `hf-chatbot/app.py` (single-file Gradio app). Features layer onto existing patterns: new UI components in `main()`, new state variables in `gr.State`, prompt modifications in `_build_messages()`, and a rewritten `_call_model()` for streaming. No new files, no new dependencies.

**Tech Stack:** Python, Gradio 4.x, HuggingFace Inference API, browser localStorage via Gradio JS interop.

---

### Task 1: System Prompt — Add Scoring + Code Block Instructions

**Files:**
- Modify: `hf-chatbot/app.py:59-102` (SYSTEM_PROMPT)

**Step 1: Update SYSTEM_PROMPT to include scoring and code instructions**

Add these two sections to the end of SYSTEM_PROMPT (before the closing `"""`):

```python
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
```

**Step 2: Verify**

Read the file and confirm the SYSTEM_PROMPT now ends with the scoring and code sections.

**Step 3: Commit**

```bash
git add hf-chatbot/app.py
git commit -m "feat: add scoring tag and code block instructions to system prompt"
```

---

### Task 2: Score Parsing Helper + Category/Resume Prompt Builders

**Files:**
- Modify: `hf-chatbot/app.py` — add new functions after `_format_duration` (~line 351), add `CATEGORY_PROMPTS` dict after `DIFFICULTY_PROMPTS` (~line 57)

**Step 1: Add CATEGORY_PROMPTS constant**

Insert after `DIFFICULTY_PROMPTS` (after line 57):

```python
CATEGORY_PROMPTS = {
    "System Design": "Focus questions on system design: scalability, distributed systems, load balancing, caching, database choices, and architecture trade-offs.",
    "Coding/Algorithms": "Focus questions on coding and algorithms: data structures, time/space complexity, common algorithms, and ask the candidate to write or trace through code.",
    "Domain Knowledge": "Focus questions on domain-specific knowledge relevant to the job description: industry concepts, tools, frameworks, and best practices mentioned in the JD.",
    "API Design": "Focus questions on API design: REST vs GraphQL, endpoint design, versioning, authentication, rate limiting, error handling, and documentation.",
    "Debugging/Troubleshooting": "Focus questions on debugging and troubleshooting: reading error messages, diagnosing production issues, systematic debugging approaches, and monitoring/observability.",
    "Architecture": "Focus questions on software architecture: design patterns, microservices vs monoliths, event-driven architecture, CQRS, and architectural decision-making.",
}
```

**Step 2: Add `_parse_score` helper**

Insert after `_format_duration` function:

```python
def _parse_score(text: str) -> tuple[str, int | None]:
    """Extract <!--SCORE:N--> from the end of a response. Returns (clean_text, score)."""
    match = re.search(r"<!--SCORE:(\d)-->", text)
    if match:
        score = int(match.group(1))
        clean = text[:match.start()].rstrip()
        return clean, max(1, min(5, score))
    return text, None
```

**Step 3: Update `_build_messages` to accept categories and resume**

Replace the existing `_build_messages` function:

```python
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
        prompt += (
            "\n\n## Candidate resume:\n"
            "The candidate has provided their resume. Use it to tailor questions to "
            "gaps between their experience and the job requirements. Here it is:\n\n"
            + resume_text
        )
    messages.append({"role": "system", "content": prompt})
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if isinstance(content, str) and role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    return messages
```

**Step 4: Update `_respond` to pass through new params**

```python
def _respond(
    message: str,
    history: list[dict],
    difficulty: str,
    categories: list[str] | None = None,
    resume_text: str = "",
) -> str:
    """Generate the next assistant reply."""
    messages = _build_messages(history, difficulty, categories, resume_text)
    messages.append({"role": "user", "content": message})
    return _call_model(messages, _get_client())
```

**Step 5: Verify**

Read the file and confirm `_parse_score`, `CATEGORY_PROMPTS`, and updated `_build_messages` / `_respond` are present.

**Step 6: Commit**

```bash
git add hf-chatbot/app.py
git commit -m "feat: add score parsing, category prompts, and resume support to message builder"
```

---

### Task 3: Streaming `_call_model`

**Files:**
- Modify: `hf-chatbot/app.py:288-323` (`_call_model` function)

**Step 1: Replace `_call_model` with a streaming generator**

Replace the existing `_call_model`:

```python
def _call_model_streaming(messages: list[dict], client: InferenceClient):
    """Send messages to the model, resolve tool calls, then stream the final response."""
    # First, resolve any tool calls (non-streaming, since we need the full tool call)
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

    # If the non-streaming response already has content (tool call resolved to final answer),
    # check if we should re-request as streaming
    if msg.content:
        yield msg.content.strip()
        return

    # Stream the final response (no more tool calls)
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
```

Also keep `_call_model` as a non-streaming wrapper for the model-answer feature:

```python
def _call_model(messages: list[dict], client: InferenceClient) -> str:
    """Non-streaming model call. Used for model-answer requests."""
    result = ""
    for partial in _call_model_streaming(messages, client):
        result = partial
    return result
```

**Step 2: Verify**

Read the file and confirm both `_call_model_streaming` and `_call_model` exist.

**Step 3: Commit**

```bash
git add hf-chatbot/app.py
git commit -m "feat: add streaming model call with tool-call resolution"
```

---

### Task 4: UI Components — Resume, Categories, Score, Model Answer, Session History

**Files:**
- Modify: `hf-chatbot/app.py:397-536` (the `main()` function)

**Step 1: Add new state variables**

After the existing `question_ts = gr.State(0.0)` line, add:

```python
scores = gr.State([])  # list of int scores
resume_state = gr.State("")  # resume text
```

**Step 2: Add category checkboxes to settings row**

Add after the difficulty Radio in the settings row:

```python
categories = gr.CheckboxGroup(
    choices=list(CATEGORY_PROMPTS.keys()),
    value=[],
    label="Focus areas (optional)",
    scale=3,
)
```

**Step 3: Add score display to settings row**

Add after timer_display:

```python
score_display = gr.Textbox(
    value="",
    label="Avg score",
    interactive=False,
    scale=1,
)
```

**Step 4: Add resume upload accordion**

Add after the JD file upload accordion:

```python
with gr.Accordion("Upload your resume (optional)", open=False):
    resume_upload = gr.File(
        label="Upload resume (PDF, DOCX, or TXT)",
        file_types=[".pdf", ".docx", ".doc", ".txt"],
    )
    resume_btn = gr.Button("Use this resume")
```

**Step 5: Add model answer button**

Add after the Send button row:

```python
with gr.Row():
    model_answer_btn = gr.Button("Show model answer", variant="secondary")
```

**Step 6: Add session history controls**

Add at the top of the UI (after the Markdown header, before settings row):

```python
with gr.Row():
    session_dropdown = gr.Dropdown(
        choices=[], label="Load past session", scale=3, interactive=True
    )
    load_session_btn = gr.Button("Load", scale=1)
    save_session_btn = gr.Button("Save session", scale=1)
```

**Step 7: Verify**

Read the `main()` function and confirm all new UI components are present.

**Step 8: Commit**

```bash
git add hf-chatbot/app.py
git commit -m "feat: add UI components for resume, categories, scoring, model answer, sessions"
```

---

### Task 5: Wire Up Handlers — Streaming Submit, Score Tracking, Resume

**Files:**
- Modify: `hf-chatbot/app.py` — rewrite `_on_text_submit`, `_on_file_upload`, `_on_voice_submit` handlers, add resume and model-answer handlers

**Step 1: Rewrite `_on_text_submit` as a streaming generator**

```python
def _on_text_submit(
    message: str, history: list[dict], diff: str, ts: float,
    score_list: list, cats: list[str], resume_txt: str,
):
    if not message.strip():
        yield history, history, "", "", ts, score_list, _fmt_avg(score_list)
        return
    duration_str = ""
    if ts > 0 and history and history[-1].get("role") == "assistant":
        elapsed = time.time() - ts
        duration_str = _format_duration(elapsed)

    history = history + [{"role": "user", "content": message}]
    messages = _build_messages(history[:-1], diff, cats, resume_txt)
    messages.append({"role": "user", "content": message})

    partial_history = history + [{"role": "assistant", "content": ""}]
    for partial_text in _call_model_streaming(messages, _get_client()):
        partial_history[-1]["content"] = partial_text
        yield (partial_history, partial_history, "", duration_str, time.time(),
               score_list, _fmt_avg(score_list))

    # Parse score from final response
    final_text = partial_history[-1]["content"]
    clean_text, score = _parse_score(final_text)
    partial_history[-1]["content"] = clean_text
    if score is not None:
        score_list = score_list + [score]
    yield (partial_history, partial_history, "", duration_str, time.time(),
           score_list, _fmt_avg(score_list))
```

Add a helper above the handlers:

```python
def _fmt_avg(scores: list) -> str:
    """Format the average score for display."""
    if not scores:
        return ""
    avg = sum(scores) / len(scores)
    return f"{avg:.1f}/5 ({len(scores)} Qs)"
```

**Step 2: Update handler wiring for text submit**

```python
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
```

**Step 3: Rewrite `_on_file_upload` similarly (streaming + score + categories + resume)**

Same pattern as `_on_text_submit` but for JD file upload. Include `cats` and `resume_txt` params.

**Step 4: Rewrite `_on_voice_submit` similarly**

Same streaming pattern for voice input.

**Step 5: Add resume upload handler**

```python
def _on_resume_upload(file):
    if file is None:
        return ""
    text = _extract_file_text(file)
    if text.startswith("Error"):
        return ""
    return text

resume_btn.click(
    _on_resume_upload,
    [resume_upload],
    [resume_state],
)
```

**Step 6: Add model answer handler**

```python
def _on_model_answer(history: list[dict], diff: str, cats: list[str], resume_txt: str):
    if not history:
        return history, history
    messages = _build_messages(history, diff, cats, resume_txt)
    messages.append({
        "role": "user",
        "content": "Give a strong, detailed model answer for the last technical question you asked. "
                   "Format it as if you were the ideal candidate responding. Use code blocks if relevant.",
    })
    reply = _call_model(messages, _get_client())
    reply = f"**Model Answer:**\n\n{reply}"
    history = history + [{"role": "assistant", "content": reply}]
    return history, history

model_answer_btn.click(
    _on_model_answer,
    [state, difficulty, categories, resume_state],
    [chatbot, state],
)
```

**Step 7: Verify**

Read handlers and confirm they all compile and wire to correct inputs/outputs.

**Step 8: Commit**

```bash
git add hf-chatbot/app.py
git commit -m "feat: wire up streaming handlers with score tracking, resume, categories, model answer"
```

---

### Task 6: Session History — localStorage Save/Load via JS

**Files:**
- Modify: `hf-chatbot/app.py` — add JS interop for session save/load in `main()`

**Step 1: Add save session handler with JS**

```python
save_session_js = """
async function(history) {
    if (!history || history.length === 0) return [];
    const key = 'interview_session_' + Date.now();
    const label = new Date().toLocaleString() + ' (' + history.length + ' msgs)';
    const data = JSON.stringify({label: label, history: history});
    localStorage.setItem(key, data);
    // Rebuild dropdown choices
    const keys = Object.keys(localStorage).filter(k => k.startsWith('interview_session_'));
    const choices = keys.map(k => {
        try { return JSON.parse(localStorage.getItem(k)).label; } catch(e) { return k; }
    });
    return choices;
}
"""

save_session_btn.click(
    None, [state], [session_dropdown], js=save_session_js
)
```

**Step 2: Add load session handler with JS**

```python
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
```

**Step 3: Add page-load JS to populate dropdown**

```python
populate_sessions_js = """
async function() {
    const keys = Object.keys(localStorage).filter(k => k.startsWith('interview_session_'));
    const choices = keys.map(k => {
        try { return JSON.parse(localStorage.getItem(k)).label; } catch(e) { return k; }
    });
    return [choices.length > 0 ? choices : []];
}
"""

demo.load(None, [], [session_dropdown], js=populate_sessions_js)
```

**Step 4: Verify**

Read the session-related code and confirm JS strings are syntactically valid.

**Step 5: Commit**

```bash
git add hf-chatbot/app.py
git commit -m "feat: add browser localStorage session save/load with dropdown"
```

---

### Task 7: Code Block Support + Final Cleanup

**Files:**
- Modify: `hf-chatbot/app.py` — Chatbot component config, placeholder text

**Step 1: Enable markdown in Chatbot**

Update the Chatbot instantiation:

```python
chatbot = gr.Chatbot(height=450, render_markdown=True)
```

**Step 2: Update placeholder text**

```python
txt = gr.Textbox(
    placeholder="Type your answer (or paste a JD / URL to begin). Code snippets supported with markdown.",
    show_label=False,
    scale=4,
)
```

**Step 3: Update the difficulty prompts to remove "behavioral" references**

The Easy and Hard difficulty prompts still mention behavioral questions — update them to stay technical-only (matching the system prompt change from earlier).

**Step 4: Verify the full app**

Run: `cd /Users/eoincoulter/Desktop/MSAI-631-Residency-G3/hf-chatbot && python -c "import app; print('OK')"`

Expected: `OK` (confirms no syntax errors)

**Step 5: Commit**

```bash
git add hf-chatbot/app.py
git commit -m "feat: enable markdown code blocks, update placeholders, clean up difficulty prompts"
```

---

## Execution Order

Tasks 1-7 are sequential — each builds on the previous. Task 1 and 2 are prompt/logic changes, Task 3 is the streaming rewrite, Tasks 4-5 are the main UI + handler work, Task 6 is the JS localStorage feature, and Task 7 is polish.
