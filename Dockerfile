FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY src ./src
COPY configs ./configs
COPY app ./app
COPY scripts ./scripts
COPY data ./data

ENV API_DB_PATH=/data/claimguard.db \
    RAG_CHROMA_DIR=/app/index/chroma \
    RAG_LIBRARY_CHROMA_DIR=/app/index/chroma_library
RUN mkdir -p /data /app/index

RUN GROQ_API_KEY=build-placeholder python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
RUN GROQ_API_KEY=build-placeholder python -m claimguard.rag.ingest
RUN GROQ_API_KEY=build-placeholder python -c "from configs.settings import load_settings; from claimguard.agent.library import prepare_library; r = load_settings().rag; prepare_library(r.library_dir, r.library_chroma_dir)"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "claimguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]