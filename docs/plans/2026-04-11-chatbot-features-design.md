# Interview Prep Bot: 7 New Features Design

## 1. Resume Upload

- New file upload accordion below the JD upload, accepts PDF/DOCX/TXT
- Resume text stored in Gradio state, injected into the system prompt as context
- Bot uses it to identify gaps between the candidate's experience and the JD requirements

## 2. Question Category Selector (Multi-select)

- Checkboxes: "System Design", "Coding/Algorithms", "Domain Knowledge", "API Design", "Debugging/Troubleshooting", "Architecture"
- Appended to the system prompt like difficulty is today
- If none selected, bot picks freely

## 3. Answer Scoring

- Model returns a JSON block at the end of each response: `{"score": 3}`
- We parse it out before displaying the message (user never sees the JSON)
- Running score tracked in state, displayed as a progress indicator in the settings row (e.g. "Avg: 3.8/5")

## 4. "Show Model Answer" Button

- Button appears in the UI below the chat input
- When clicked, sends a hidden prompt: "Give a strong model answer for the last question you asked"
- Response appended as a new assistant message

## 5. Session History (Browser Local Storage)

- On session end or export, serialize chat history + metadata to JSON and save via Gradio's localStorage JS interop
- Dropdown at the top to load a past session, populated on page load via JS
- Sessions keyed by timestamp + JD snippet

## 6. Streaming Responses

- Switch `_call_model` to use `stream=True`
- Yield partial tokens back through Gradio's streaming chatbot interface
- Tool calls still resolved synchronously before streaming the final response

## 7. Code Block Support

- Gradio Chatbot with markdown rendering enabled
- Placeholder text notes that code snippets are supported
- System prompt tells the model to use markdown code blocks when showing code

## Constraints

- All changes in `app.py` (single-file app)
- No new dependencies beyond what's already in requirements.txt
- Browser local storage for sessions (no server-side persistence)
