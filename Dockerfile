# ============================================================
# DataPilot AI Pro — Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 — build (install deps)
#   Stage 2 — runtime (slim image)
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System deps for kaleido & plotly image export
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd --gid 1000 datapilot && \
    useradd --uid 1000 --gid datapilot --create-home datapilot

WORKDIR /app

# Copy project
COPY . .

# Output directory
RUN mkdir -p /app/output && chown -R datapilot:datapilot /app

USER datapilot

# Streamlit ports
EXPOSE 8501
# FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: Streamlit
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
