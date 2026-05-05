FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST — prevents pip from pulling 3GB of CUDA packages
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Fix line endings on start.sh (in case of Windows CRLF)
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["./start.sh"]
