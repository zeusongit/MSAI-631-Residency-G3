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
    """Send messages to the model, resolve tool calls, then stream the final response.

    Streams directly from the first request. If the model returns tool calls
    instead of content, resolves them and then streams the follow-up.
    """
    # Try streaming first — this is the common path (no tool calls)
    stream = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=MAX_NEW_TOKENS,
        stream=True,
    )

    # Collect the stream, watching for tool calls
    partial = ""
    tool_calls_raw: list[dict] = {}  # index -> {id, name, arguments}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Accumulate streamed content
        if delta.content:
            partial += delta.content
            yield partial

        # Accumulate tool call fragments
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_raw:
                    tool_calls_raw[idx] = {
                        "id": tc.id or "",
                        "name": tc.function.name if tc.function and tc.function.name else "",
                        "arguments": "",
                    }
                if tc.id:
                    tool_calls_raw[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_raw[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_raw[idx]["arguments"] += tc.function.arguments

    # If we got content and no tool calls, we're done
    if partial and not tool_calls_raw:
        return

    # If no tool calls were found at all, return whatever we have
    if not tool_calls_raw:
        return

    # Resolve tool calls
    assembled_tool_calls = []
    for idx in sorted(tool_calls_raw.keys()):
        tc_data = tool_calls_raw[idx]
        assembled_tool_calls.append({
            "id": tc_data["id"],
            "type": "function",
            "function": {"name": tc_data["name"], "arguments": tc_data["arguments"]},
        })

    messages.append({"role": "assistant", "tool_calls": assembled_tool_calls})

    for tc_dict in assembled_tool_calls:
        # Create a simple object to pass to execute_tool
        class _ToolCall:
            def __init__(self, d):
                self.id = d["id"]
                self.function = type("F", (), {
                    "name": d["function"]["name"],
                    "arguments": d["function"]["arguments"],
                })()
        messages.append(execute_tool(_ToolCall(tc_dict)))

    # After tool resolution, check if model wants more tools or is ready to respond
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=MAX_NEW_TOKENS,
    )
    msg = response.choices[0].message

    # If more tool calls, resolve them iteratively
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

    # Stream the final response after all tools are resolved
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
