"""Chat / LLM logic — message building, model calls, and response generation."""

from __future__ import annotations

from huggingface_hub import InferenceClient

from config import (
    CATEGORY_PROMPTS,
    DIFFICULTY_PROMPTS,
    HF_TOKEN,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
)
from tools import TOOLS, execute_tool


def build_messages(
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


def call_model_streaming(messages: list[dict], client: InferenceClient):
    """Send messages to the model, resolve tool calls, then stream the final response."""
    # First, do a non-streaming call to check if the model wants to use tools.
    # We need the full response to see tool_calls.
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=MAX_NEW_TOKENS,
    )
    msg = response.choices[0].message

    # If no tool calls, re-request as streaming for real-time token output
    if not msg.tool_calls:
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            stream=True,
        )
        partial = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                partial += chunk.choices[0].delta.content
                yield partial
        return

    # Resolve tool calls (non-streaming since we need complete tool results)
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
            messages.append(execute_tool(tc))
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=MAX_NEW_TOKENS,
        )
        msg = response.choices[0].message

    # After tool resolution, stream the final response
    stream = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        stream=True,
    )
    partial = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial


def call_model(messages: list[dict], client: InferenceClient) -> str:
    """Non-streaming model call. Used for model-answer requests."""
    result = ""
    for partial in call_model_streaming(messages, client):
        result = partial
    return result


_client: InferenceClient | None = None


def get_client() -> InferenceClient:
    """Lazy-initialise the HF Inference client."""
    global _client
    if _client is None:
        _client = InferenceClient(token=HF_TOKEN)
    return _client


def respond(
    message: str,
    history: list[dict],
    difficulty: str,
    categories: list[str] | None = None,
    resume_text: str = "",
) -> str:
    """Generate the next assistant reply."""
    messages = build_messages(history, difficulty, categories, resume_text)
    messages.append({"role": "user", "content": message})
    return call_model(messages, get_client())
