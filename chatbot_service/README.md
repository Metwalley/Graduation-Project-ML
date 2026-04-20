# `chatbot_service/` — RAG Educational Chatbot

Internal FastAPI microservice powering the educational chatbot. **Not exposed publicly** — the `api/` engine proxies to it.

Runs on port `8001` (internal Docker network only).

## How It Works

```
Question → Safety Filter → FAISS Article Search → Llama 3.2 (Ollama) → Arabic Response
```

| File | Role |
|---|---|
| `main.py` | FastAPI app, `/chat` and `/health` endpoints |
| `rag_engine.py` | Orchestrates the 4-step RAG pipeline |
| `safety_filter.py` | Blocks medical advice and prescription requests |
| `knowledge_base.py` | Loads articles, builds FAISS vector index |
| `ollama_client.py` | Calls local Llama 3.2 with Arabic system prompt |
| `response_formatter.py` | Post-processing cleanup |
| `config.py` | Settings loaded from environment variables |

## Key Design Decisions

- **Fully offline** — No API keys, no internet required after first model pull
- **Arabic-only** — System prompt + post-processing filter enforce Arabic output
- **Temperature = 0.1** — Low, for consistent factual responses

## Run Locally

```bash
# Requires Ollama running first: ollama serve
cd chatbot_service
$env:TF_ENABLE_ONEDNN_OPTS="0"
uvicorn main:app --port 8001
```
