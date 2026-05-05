#!/bin/bash
# start.sh - Entry point for chatbot-service container

echo "Starting Chatbot Service Container..."
echo "Waiting for Ollama at $OLLAMA_BASE_URL ..."

until curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null; do
  echo "Ollama not ready yet, retrying in 5s..."
  sleep 5
done

echo "Ollama is UP!"

echo "Checking for model: $OLLAMA_MODEL ..."
if curl -sf "$OLLAMA_BASE_URL/api/tags" | grep -q "$OLLAMA_MODEL"; then
  echo "Model $OLLAMA_MODEL already exists."
else
  echo "Pulling $OLLAMA_MODEL (this may take a while)..."
  curl -X POST "$OLLAMA_BASE_URL/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$OLLAMA_MODEL\"}"
  echo "Model pulled."
fi

if [ "$APP_MODE" = "api" ]; then
    echo "Starting FastAPI on port 8001..."
    exec uvicorn main:app --host 0.0.0.0 --port 8001
else
    echo "Starting Streamlit UI on port 8501..."
    exec streamlit run chat_ui.py --server.port 8501 --server.address 0.0.0.0
fi
