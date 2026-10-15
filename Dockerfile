# ==============================================================================
# Dockerfile for Multimodal RAG System
# Supports Text, Code, Image (OCR), Audio, and Video processing
# ==============================================================================

FROM python:3.12-slim

# Avoid prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install required system packages for OpenCV, FFmpeg, and PyAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY backend/requirements.txt /app/backend/requirements.txt

# Install PyTorch & core dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r backend/requirements.txt && \
    pip install --no-cache-dir easyocr soundfile av librosa

# Copy application source code
COPY backend /app/backend
COPY frontend /app/frontend
COPY data /app/data

# Ensure data and uploads directories exist
RUN mkdir -p /app/data/uploads /app/data/lancedb

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["python3", "backend/run_server.py"]
