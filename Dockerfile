# CPU image by default. For NVIDIA GPUs: docker build --build-arg TORCH_INDEX=cu128 -t nmt-uz:cuda .
FROM python:3.11-slim

ARG TORCH_INDEX=cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/cache/huggingface \
    MODEL_ID=nickoo004/nmt-uz-76m

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache "torch>=2.4" \
        --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
 && uv pip install --system --no-cache \
        "sentencepiece>=0.2" "sacrebleu>=2.4" "polars>=2.1" safetensors huggingface_hub \
        fastapi "uvicorn[standard]"

WORKDIR /app
COPY src/ src/
RUN useradd --create-home --uid 1000 app && mkdir -p /cache && chown app /cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
