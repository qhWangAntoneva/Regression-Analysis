# ===========================================================================
# Dockerfile — Regression Analysis Streamlit App
# Multi-stage build with uv for fast dependency resolution
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only dependency files first — better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual env (no dev deps needed at runtime)
RUN uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy uv-installed venv from builder
COPY --from=builder /app/.venv /app/.venv

# Set virtual env as default Python
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY app/ ./app/
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY pyproject.toml ./

# Install the project itself into venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
RUN uv pip install --no-deps -e .

# Create data directory for uploads/exports
RUN mkdir -p /app/data

EXPOSE 8501

# Run Streamlit (no file watcher needed in prod)
CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
