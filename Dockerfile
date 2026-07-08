FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first, so sentence-transformers (a requirements.txt dep)
# finds its torch>=1.11.0 requirement already satisfied and skips pulling in
# the much larger default CUDA/nvidia build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

EXPOSE 8000

CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
