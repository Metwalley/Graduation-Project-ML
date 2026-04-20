#!/bin/bash
# start.sh - Entry point script

echo "🚀 Starting Chatbot Service Container..."

# Note: In docker-compose, we have a separate 'ollama' service.
# We don't run ollama inside this python container.
# We connect to the ollama service via network.

echo "⏳ Waiting for Ollama service to be ready at $OLLAMA_BASE_URL..."

# Wait loop for Ollama
# We check the /api/tags endpoint to see if service is up
until curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null; do
  echo "zzz... Waiting for Ollama..."
  sleep 5
done

echo "✅ Ollama is UP!"

# Check if model is pulled, if not pull it
# This uses the volume shared with the ollama service
echo "📦 Checking for model: $OLLAMA_MODEL..."
if curl -s "$OLLAMA_BASE_URL/api/tags" | grep -q "$OLLAMA_MODEL"; then
  echo "✅ Model $OLLAMA_MODEL already exists."
else
  echo "⬇️ Model not found. Pulling $OLLAMA_MODEL (this may take a while)..."
  # We use curl to trigger the pull on the ollama service
  curl -X POST "$OLLAMA_BASE_URL/api/pull" -d "{\"name\": \"$OLLAMA_MODEL\"}"
  echo "✅ Model pulled successfully."
fi

# Start the application
# We can start either the API (FastAPI) or Streamlit UI based on env var
if [ "$APP_MODE" = "api" ]; then
    echo "🌐 Starting FastAPI Server..."
    uvicorn main:app --host 0.0.0.0 --port 8001
else
    echo "📺 Starting Streamlit UI..."
    streamlit run chat_ui.py --server.port 8501 --server.address 0.0.0.0
fi
