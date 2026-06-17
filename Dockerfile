FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit jiwer

# Copy application source
COPY src/       ./src/
COPY app.py     .
COPY .streamlit/ ./.streamlit/

# Persistent directories
RUN mkdir -p /app/output /app/videos

# Non-root user for security
RUN useradd -m -u 1000 whisper && chown -R whisper:whisper /app
USER whisper

# Hugging Face model cache persists via Docker volume
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
