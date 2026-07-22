# API image for the too-good-to-be-true-backtester backend (FastAPI).
# Build context is the repo root so the tgtbt package installs alongside the api/ app.
FROM python:3.12-slim

WORKDIR /app

# Install the tgtbt package (pulls its own deps from pyproject) plus the web-serving stack.
COPY pyproject.toml README.md ./
COPY tgtbt ./tgtbt
RUN pip install --no-cache-dir . "fastapi>=0.110" "uvicorn[standard]>=0.29"

COPY api ./api

ENV PORT=8000
EXPOSE 8000
# Hosts inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
