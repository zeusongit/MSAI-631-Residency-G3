Here is the proposal in markdown format — copy and paste it into a .md file:

# Group Project Proposal
## AI-Powered Interview Preparation Chatbot
**Course:** MSAI-631 — Human-Computer Interaction
**Date:** April 2026
---
## 1. Problem Statement
Job seekers, particularly those entering or transitioning into technical fields,
face a significant challenge in preparing for interviews. Traditional preparation
methods — reading guides, mock interviews with peers, or hiring coaches — are
either time-consuming, expensive, or unavailable on demand. There is a clear need
for an accessible, intelligent, and interactive tool that can simulate real
interview scenarios and provide immediate, personalized feedback.
This project proposes an AI-powered **Interview Preparation Chatbot** that acts
as an on-demand mock interviewer. Given a job description, the chatbot generates
tailored technical and behavioral questions, evaluates the user's responses, and
provides constructive feedback — all through a conversational interface.
---
## 2. Proposed Solution
The system will be a conversational agent built on a large language model (LLM)
accessed through Hugging Face's Inference API. The chatbot will:
- Accept a **job description** (pasted text) as context
- Generate **role-specific interview questions** (technical + behavioral)
- Evaluate user answers and provide **structured feedback**
- Ask **follow-up questions** to simulate a real interview flow
- Produce an end-of-session **performance summary**
The interaction is entirely conversational, making it a direct example of
AI-driven Human-Computer Interaction.
---
## 3. Scope
### In Scope
- Single-turn and multi-turn conversation for interview Q&A
- Support for software engineering and data science role types
- Text-based input and output via a Gradio web UI
- Session summary at the end of the interview
### Out of Scope
- Real-time voice interaction (stretch goal only)
- Resume parsing or document uploads
- Integration with job boards or external databases
- User accounts or persistent history across sessions
---
## 4. Tech Stack
| Layer | Technology | Justification |
|---|---|---|
| **UI** | [Gradio](https://gradio.app) | Python-native, minimal setup, free HF Spaces support |
| **LLM** | `mistralai/Mistral-7B-Instruct-v0.3` via HF Inference API | Free tier, strong instruction following, no local GPU needed |
| **Backend** | Python 3.10+ | Cross-platform, widely used in ML |
| **Hosting** | Hugging Face Spaces (CPU free tier) | Free, shareable, no infrastructure management |
| **Versioning** | GitHub | Collaboration, PR workflow, instructor access |
### Model Rationale
`mistralai/Mistral-7B-Instruct-v0.3` is publicly available on Hugging Face,
accessible via the free Inference API, and produces high-quality instruction-
following outputs — sufficient for generating interview questions and evaluating
answers without requiring local GPU resources.
---
## 5. Development Platform
- **Code:** Python, hosted in a GitHub repository
- **Deployment:** Hugging Face Space (Gradio SDK, CPU free tier)
- **Version Control:** GitHub for source; HF Space as deployment target
- **Local Development:** Any laptop with Python 3.10+ and a virtual environment
---
## 6. Candidate Models
| Model | Size | Notes |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | **Primary choice** — free API, strong quality |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Fallback for fully local, CPU-only execution |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Alternate if Mistral quota is exceeded |
---
## 7. Anticipated Limitations
- **Response latency** on free HF Inference API may be 5–15 seconds per turn
- **Context window** limits may truncate long job descriptions
- **Model quality** at 7B parameters may occasionally produce generic questions
- No persistent memory between sessions
---
## 8. References
- Hugging Face Gradio Docs: https://www.gradio.app/docs
- Hugging Face Inference API: https://huggingface.co/docs/api-inference
- KDNuggets Chatbot Tutorial: https://www.kdnuggets.com/2023/06/build-ai-chatbot-5-minutes-hugging-face-gradio.html
- Mistral 7B Model Card: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3