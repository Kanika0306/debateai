# =============================================================================
# backend/Dockerfile — Debate-AI FastAPI backend (DEV configuration)
# =============================================================================
FROM python:3.11-slim

# ---- System dependencies ----
# ffmpeg: required at RUNTIME by faster-whisper and pyannote.audio for audio
#         decoding. Missing ffmpeg causes silent runtime failures, not build
#         errors, so it's easy to miss during testing.
# libsndfile1: required by soundfile/librosa for reading audio formats.
# gcc/g++: needed to compile some Python C extensions (e.g. webrtcvad).
# git: needed by some pip packages that pull from git repos.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies (layer-cached) ----
# Copy requirements.txt FIRST so Docker can cache this expensive pip install
# layer. If only application code changes (not requirements.txt), rebuilds
# reuse the cached layer instead of reinstalling everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Install PyTorch CPU (for Docker — keeps image smaller than GPU build) ----
# Using CPU-only torch to keep image size manageable. For GPU support inside
# Docker, switch to a CUDA base image and the appropriate torch index URL.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# ---- Application code ----
# Copied after pip install so code-only changes don't invalidate the pip cache.
# In dev mode, these are overridden by the bind mount in docker-compose.yml anyway.
COPY . .

EXPOSE 8000

# ---- Entrypoint ----
# DEV ONLY: --reload watches for file changes and auto-restarts.
# PRODUCTION: remove --reload and consider using gunicorn with uvicorn workers:
#   CMD ["gunicorn", "backend.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
