# Build multi-service production container for CFPB RAG Intelligence
FROM python:3.12-slim

# Set environment invariants
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install basic OS utilities and C++ build tools for compiled ML wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker build layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source directories
COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY evals/ ./evals/

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command launches the FastAPI production backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

