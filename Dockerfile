# syntax=docker/dockerfile:1

# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────────
# gcc needed for some sentence-transformers / chromadb C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
# Copy only the files needed for dependency installation first (layer caching)
COPY pyproject.toml ./

# Install the project and all its dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e .

# ── Application source and data ──────────────────────────────────────────────
COPY src/ ./src/
COPY data/ ./data/

# ── Build the Chroma vector store ────────────────────────────────────────────
# Runs the ingest step at image build time so the Chroma database is baked
# into the image. Retrieval works immediately on the deployed container with
# no runtime setup and no Ollama — sentence-transformers embeds on CPU only.
RUN python -m src.ingest

# ── Runtime configuration ─────────────────────────────────────────────────────
# Expose the default port (Render overrides with $PORT at runtime)
EXPOSE 8000

# LLM_PROVIDER and GROQ_API_KEY must be set in the host dashboard as secrets.
# They are NOT baked into this image.
ENV LLM_PROVIDER=groq

# ── Start the API ─────────────────────────────────────────────────────────────
# Read PORT from the environment (Render sets this dynamically).
# Default to 8000 when running locally.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
