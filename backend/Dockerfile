# ═══════════════════════════════════════════════════════════════
# QAD-MultiGuard v4.1 — Production multi-stage Docker build
# Soft-edge collaborative multimodal detection
# ═══════════════════════════════════════════════════════════════

ARG PYTHON_VERSION=3.11-slim

# ── Stage 1: Dependency installation ──────────────────────────
FROM python:${PYTHON_VERSION} AS builder

WORKDIR /build

ARG REQUIREMENTS=requirements.txt
COPY ${REQUIREMENTS} requirements.txt

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime image ────────────────────────────────────
FROM python:${PYTHON_VERSION}

LABEL maintainer="Campus Safety Team"
LABEL version="4.1.0"
LABEL description="QAD-MultiGuard API Server"

WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application source (excludes are handled by .dockerignore)
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
