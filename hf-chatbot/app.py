"""
Minimal Gradio chat app using a small instruct model from the Hugging Face Hub.
Runs on CPU by default (suitable for laptops and free CPU Spaces).

Set HF_MODEL_ID to swap models. For gated models, set HUGGING_FACE_HUB_TOKEN
in the environment (never commit tokens).
"""

from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

MODEL_ID = os.environ.get("HF_MODEL_ID", DEFAULT_MODEL_ID)
MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", "256"))
SYSTEM_PROMPT = os.environ.get(
    "HF_SYSTEM_PROMPT",
    "You are a helpful, concise assistant. If you are unsure, say you are unsure.",
)


def _load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


TOKENIZER, MODEL = _load_model_and_tokenizer()


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Gradio 5.x multimodal format: [{"type": "text", "text": "..."}, ...]
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def _build_messages(user_message: str, history: list) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
    for item in history:
        # Gradio 5.x passes history as list of dicts with "role"/"content" keys
        if isinstance(item, dict):
            role = item.get("role", "")
            content = _extract_text(item.get("content", ""))
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        else:
            # Gradio 4.x passed [user_text, assistant_text] pairs
            user_text, assistant_text = item
            messages.append({"role": "user", "content": _extract_text(user_text)})
            if assistant_text:
                messages.append({"role": "assistant", "content": _extract_text(assistant_text)})
    messages.append({"role": "user", "content": user_message})
    return messages


def respond(message: str, history: list[list[str | None]]) -> str:
    messages = _build_messages(message, history)
    input_text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = TOKENIZER(input_text, return_tensors="pt")
    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    demo = gr.ChatInterface(
        fn=respond,
        title="HF Chat (local / Space)",
        description=f"Model: `{MODEL_ID}`. CPU inference — first reply may take a bit.",
    )
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))


if __name__ == "__main__":
    main()
