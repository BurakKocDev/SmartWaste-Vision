FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SMARTWASTE_MODEL_PATH=/models/best_model.pth \
    SMARTWASTE_TORCH_THREADS=4

WORKDIR /srv/smartwaste

RUN python -m pip install --no-cache-dir --upgrade pip==26.2 && \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.12.1+cpu torchvision==0.27.1+cpu

COPY requirements-api.in ./requirements-api.in
RUN python -m pip install --no-cache-dir \
      --index-url https://pypi.org/simple \
      -r requirements-api.in

COPY api ./api
COPY artifacts/training/mobilenet_v3_small_baseline/best_model.pth /models/best_model.pth

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /srv/smartwaste /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
