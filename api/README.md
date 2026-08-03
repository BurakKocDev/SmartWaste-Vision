# SmartWaste Vision PyTorch API

The small reference service exposes the FP32 PyTorch checkpoint through two
endpoints:

- `GET /health`
- `POST /v1/classify` with a `file` multipart field containing JPEG or PNG

It is intentionally not used by the Flutter app; mobile inference remains fully
offline. The service exists to demonstrate the server deployment of the same
checkpoint and to provide a reproducible reference output.

## Local run

```bash
bash scripts/setup_api_env.sh
SMARTWASTE_MODEL_PATH=artifacts/training/mobilenet_v3_small_baseline/best_model.pth \
  ~/.cache/smartwaste-vision/api-venv/bin/uvicorn api.app.main:app \
  --host 0.0.0.0 --port 8000
```

## Docker

From the repository root:

```bash
docker build -t smartwaste-vision-api .
docker run --rm -p 8000:8000 smartwaste-vision-api
```

Example:

```bash
curl -F "file=@waste.jpg;type=image/jpeg" http://localhost:8000/v1/classify
```

The image limit is 10 MiB and 25 megapixels. The API runs one dominant object
through `MobileNetV3-Small`; it is not object detection and does not accept
multiple files.

## Verified smoke test

The runtime-only image was built and exercised on 3 August 2026 with a real
RealWaste JPEG:

- image tag: `smartwaste-vision-api:local`
- image size: 330,941,526 bytes (315.61 MiB)
- `/health`: HTTP 200
- `/v1/classify`: HTTP 200, predicted `plastic`
- confidence: `0.4824`, therefore `uncertain: true` at the `0.51` threshold
- model-only inference reported by the request: `18.44 ms`
- one-container memory snapshot after the request: `253 MiB`

The latency and memory values above are environment-specific smoke
observations, not a controlled benchmark. Use the documented CPU benchmark for
runtime comparisons and measure the container again on the target host before
capacity planning.
