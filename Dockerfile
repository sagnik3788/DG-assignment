FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 DATA_PATH=/data/documents.jsonl FASTEMBED_CACHE_PATH=/opt/fastembed_cache FASTEMBED_INDEX_PATH=/opt/fastembed_index.npz
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data ./data
COPY README.md ./README.md
RUN mkdir -p /data "$FASTEMBED_CACHE_PATH" && if [ -s /app/data/documents.jsonl ]; then cp /app/data/documents.jsonl /data/documents.jsonl; fi && if [ -s /app/data/listings.jsonl ]; then cp /app/data/listings.jsonl /data/listings.jsonl; fi && python -c "import json; from pathlib import Path; from app.retrieval import warm_retrieval_index; docs=[json.loads(x) for x in Path('/data/documents.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; assert warm_retrieval_index(docs)"
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
