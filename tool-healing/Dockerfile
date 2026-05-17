FROM python:3.11-slim

# Environment for reliable, quiet Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy application code
COPY . .

EXPOSE 8081

# Run the proxy. Allow overriding via env vars without changing the image.
# Defaults mirror docker-compose wiring and local usage.
CMD ["sh", "-c", "python __main__.py \
  --upstream ${UPSTREAM_URL:-http://llama-swap:8080} \
  --timeout ${TIMEOUT:-600} \
  --host ${HOST:-0.0.0.0} \
  --port ${PORT:-8081}"]
