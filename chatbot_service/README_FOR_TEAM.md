# 🐳 For Team: How to Run the Chatbot Service

## 🚀 Quick Start (One Command)
If you have Docker installed, just run:

```bash
docker-compose up --build
```

**That's it.** 
- ⏳ The first time might take a few minutes to download the Llama 3.2 model (2GB).
- ✅ Once running, the Chatbot is ready.

---

## 🔌 Integration Points

### 1. Interactive UI (For Demo/Testing)
- **URL:** `http://localhost:8501`
- **What it is:** A web interface to chat directly with the bot.

### 2. API Endpoint (For Backend Integration)
- **Base URL:** `http://localhost:8001`
- **Chat Endpoint:** `POST /chat`
- **Payload:**
  ```json
  {
    "text": "User question here"
  }
  ```
- **Response:**
  ```json
  {
    "response": "Chatbot answer...",
    "sources": [...]
  }
  ```

---

## 🛠️ Troubleshooting

### "It's slow!"
- **Cause:** Running on CPU only.
- **Fix:** If you have an NVIDIA GPU, uncomment the `deploy` section in `docker-compose.yml`.

### "Ollama connection error"
- **Check:** Ensure the `ollama` container is healthy.
- **Command:** `docker logs chatbot-ollama`

---

## 📂 Project Structure (For Curiosity)
- `Dockerfile`: Builds the Python environment.
- `docker-compose.yml`: Orchestrates Python app + Ollama service.
- `start.sh`: Smart script that waits for Ollama and pulls the model automatically.
