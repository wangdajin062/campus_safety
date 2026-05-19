# ═══════════════════════════════════════════════════════════════
# QAD-MultiGuard v4.1 — 多阶段构建
# 软硬协同多模态检测 | 推测解码 | 隐私保护
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: 依赖安装 ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: 运行镜像 ──────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Campus Safety Team"
LABEL version="4.1.0"
LABEL description="QAD-MultiGuard — Speculative Decoding + QAD-4bit + Multimodal Fusion"

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制依赖
COPY --from=builder /install /usr/local

# 复制源码
COPY . .

# 非 root 用户运行
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
