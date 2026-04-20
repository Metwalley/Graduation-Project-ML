# `api/` — Unified AI Engine

The core FastAPI service. This is the **only thing the Spring Boot backend needs to talk to**.

Runs on port `8000`.

## Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Service status + loaded models |
| `POST` | `/predict` | Initial diagnosis (Autism / ADHD / Dyslexia) |
| `POST` | `/monthly-tracker` | Monthly progress scoring |
| `GET` | `/monthly-tracker/questions/{disorder}` | Fetch questionnaire for Flutter |
| `POST` | `/chat` | RAG chatbot (proxied internally to chatbot-service) |

## Key Files

- **`main.py`** — All logic: ML loading, diagnosis endpoints, monthly tracker scoring engine, chat proxy
- **`ml_models/`** — Trained `.joblib` files loaded at startup (do not modify)
- **`Dockerfile`** — Used by `docker-compose`
- **`requirements.txt`** — Python dependencies

## Run Locally

```bash
cd api
pip install -r requirements.txt
$env:CHATBOT_SERVICE_URL="http://localhost:8001"
uvicorn main:app --port 8000
```

Interactive docs: `http://localhost:8000/docs`
