FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/       ./src/
COPY app.py     .
COPY .streamlit/ ./.streamlit/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create user and set ownership on build-time dirs
RUN useradd -m -u 1000 whisper && \
    mkdir -p /app/output /app/videos /app/.cache/huggingface && \
    chown -R whisper:whisper /app

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint runs as root to fix volume permissions, then drops to whisper
ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
