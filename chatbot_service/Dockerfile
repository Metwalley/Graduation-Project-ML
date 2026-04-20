# Dockerfile for Chatbot Service
# Optimized to keep image size reasonable
# The model will be downloaded at runtime to volume, not baked into image

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl is needed for Ollama checks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose API port
EXPOSE 8001
# Expose Streamlit port (optional, for demo)
EXPOSE 8501

# Scripts to run on container start
# We will use a startup script to ensure Ollama is ready before app starts
COPY start.sh .
RUN chmod +x start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["./start.sh"]
