# SmartWaste Vision

English · [Türkçe](README.md)

SmartWaste Vision classifies one dominant waste object from a photo and deploys
the same trained model across PyTorch, ONNX Runtime, and LiteRT/TensorFlow Lite.
The project compares accuracy, Macro F1, model size, latency, and memory while
keeping the Android user experience fully offline.

> Portfolio claim: I deployed the same image classifier to server, desktop CPU,
> and Android edge environments and compared its accuracy, latency, memory, and
> storage trade-offs.

## Status

- Date: 4 August 2026
- Data audit and deterministic split: complete
- MobileNetV3-Small transfer learning and frozen test evaluation: complete
- ONNX FP32, LiteRT FP16, and LiteRT INT8 weight-only exports: accepted
- Full-INT8 PTQ exports: rejected by the quality gate, not deployed
- Modern Flutter camera/gallery UI plus debug and release-mode test APKs: complete; Android emulator smoke tests passed
- Dockerised PyTorch/FastAPI reference service: verified with a real JPEG
- Final demo video: complete
- Remaining: physical Android latency/RAM run

## Mobile interface

| Home | Prediction result |
|---|---|
| ![SmartWaste Vision home screen](docs/assets/mobile-home-v2.png) | ![SmartWaste Vision prediction result](docs/assets/mobile-result-v2.png) |

The interface exposes offline status, a focused capture surface, class and bin
guidance, the top three probabilities, and a separate low-confidence recovery
experience.

[Watch the final demo video](docs/assets/smartwaste-vision-demo-tr.mp4)

## MVP scope

- Six classes: plastic, glass, metal, paper/cardboard, organic, general waste
- One ImageNet-pretrained `MobileNetV3-Small`, fine-tuned with PyTorch
- PyTorch FP32, ONNX FP32, LiteRT FP16, and INT8 storage experiments
- Accuracy, Macro F1, size, mean/p95 latency, and process memory comparison
- Android-first Flutter app with camera/gallery, confidence, bin colour,
  abstention below `0.51`, and fully offline inference
- Small Dockerised PyTorch/FastAPI reference API

Object detection, segmentation, live conveyor video, maps, rewards, accounts,
and an admin panel are explicitly out of scope.

## Results

Frozen 598-image test split:

| Runtime | Accuracy | Macro F1 | Size | Mean / p95 CPU |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 90.80% | 0.9093 | 5.95 MiB | 5.24 / 6.17 ms |
| ONNX FP32 | 90.80% | 0.9093 | 6.08 MiB | 1.35 / 1.79 ms |
| LiteRT FP16 | 90.64% | 0.9074 | 3.07 MiB | 6.80 / 8.11 ms |
| LiteRT INT8 weight-only | 89.80% | 0.8989 | 1.75 MiB | 0.67 / 0.95 ms |

CPU latency was measured on an Intel i5-11400H under WSL2, batch 1, four
threads, excluding decode and preprocessing. The Flutter model with actual Dart
preprocessing reaches 90.47% Accuracy and 0.9056 Macro F1. Weight-only INT8
stores quantised weights but still uses float input/output and compute; it is not
claimed as full-integer inference. See the
[benchmark protocol and results](docs/benchmarks/02-cpu-baseline-results.md).

![Test confusion matrix](docs/assets/confusion_matrix.svg)

## Application behaviour

The Flutter client embeds the accepted FP16 model. It takes a photo or gallery
image, performs Pillow-compatible antialiased preprocessing in Dart, displays
the class and softmax confidence, and maps the result to the Turkish Zero Waste
bin colours. Scores below `0.51` are shown as “I'm not sure.” Images never leave
the device.

The Docker API is deliberately separate. It demonstrates server deployment of
the same FP32 checkpoint through `GET /health` and multipart
`POST /v1/classify`.

## Quick start

Flutter tests plus debug and release-mode test APKs:

```powershell
cd apps/mobile
flutter pub get
flutter test
cd ../..
.\scripts\build_android.ps1 -Configuration Debug
.\scripts\build_android.ps1 -Configuration Release
```

The release-mode test APK is written to
`artifacts/mobile/smartwaste-vision-release.apk`. It is currently signed with a
debug key for testing and is not a Play Store distribution artifact.

Reference API:

```bash
docker build -t smartwaste-vision-api .
docker run --rm -p 8000:8000 smartwaste-vision-api
curl -F "file=@waste.jpg;type=image/jpeg" http://localhost:8000/v1/classify
```

## Documentation

- [Model Card](docs/model/MODEL_CARD.en.md)
- [Dataset Card](docs/data/DATASET_CARD.en.md)
- [Baseline and export accuracy](docs/model/01-baseline-results.md)
- [CPU benchmark](docs/benchmarks/02-cpu-baseline-results.md)
- [Flutter MVP and preprocessing validation](docs/mobile/01-flutter-mvp.md)
- [INT8 deployment decision](docs/decisions/ADR-002-int8-deployment-policy.md)
- [PyTorch API](api/README.md)
- [Portfolio release checklist (Turkish)](docs/release/01-portfolio-release-checklist.md)
- [Implementation plan](PROJECT_PLAN.md)

## Limitations

The model expects one clear, dominant object. It is not disposal authority and
must not be used for hazardous, medical, or electronic waste decisions. The
training data comes from one Australian facility, plastic is the weakest class,
and independent phone-captured field evaluation is still required.

RealWaste license records conflict: UCI metadata says CC BY 4.0 while the
downloaded and owner READMEs say CC BY-NC-SA 4.0. This project conservatively
assumes the more restrictive terms; public/commercial weight distribution needs
a separate review. See the [Dataset Card](docs/data/DATASET_CARD.en.md).

## Next gate

Install the release-mode APK on a physical Android phone, verify camera/gallery
and airplane-mode inference, then record model latency, end-to-end mean/p95
latency, and peak RAM. The redesigned-interface demo is already included in the
repository.
