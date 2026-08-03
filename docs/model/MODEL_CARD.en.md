# Model Card — SmartWaste Vision MobileNetV3-Small

[Türkçe sürüm](MODEL_CARD.md)

## Model summary

SmartWaste Vision classifies the single dominant waste object in one image into
six material categories. An ImageNet-pretrained `MobileNetV3-Small` was
fine-tuned on RealWaste with PyTorch.

- Version: MVP baseline, 3 August 2026
- Classes: `plastic`, `glass`, `metal`, `paper_cardboard`, `organic`, and
  `general_waste`
- Data: 2,800 train / 600 validation; a frozen 598-image test split
- Input: RGB, short side resized to 256, `224×224` center crop, ImageNet mean/std
- Output: six logits; the UI displays the top-1 softmax confidence
- Mobile abstention threshold: `0.51`; lower scores produce “I'm not sure”

## Intended use

The model is an educational and portfolio-grade waste sorting aid for a clear,
single object. It runs fully offline in the Flutter app and as PyTorch FP32 in a
small reference API.

It is not designed for multi-object detection, hazardous/medical/electronic
waste decisions, binding municipality-specific disposal guidance, live
industrial conveyor video, or unsupervised operational decisions.

## Performance

Checkpoint selection used validation Macro F1 only; the test split was not used
for selection.

| Runtime | Test Accuracy | Macro F1 | Decision |
|---|---:|---:|---|
| PyTorch FP32 | 90.80% | 0.9093 | Reference |
| ONNX FP32 | 90.80% | 0.9093 | Accepted |
| LiteRT FP16, Python/Pillow preprocessing | 90.64% | 0.9074 | Accepted |
| LiteRT FP16, actual Dart preprocessing | 90.47% | 0.9056 | Flutter model |
| LiteRT INT8 weight-only | 89.80% | 0.8989 | Accepted; float compute |
| ONNX full INT8 PTQ | 24.08% | 0.0823 | Rejected |
| LiteRT full INT8 PTQ | 10.54% | 0.0318 | Rejected |

PyTorch FP32 test F1 by class: plastic `0.843`, glass `0.930`, metal `0.902`,
paper/cardboard `0.959`, organic `0.929`, and general waste `0.894`. See the
[confusion matrix](../assets/confusion_matrix.svg) and the full
[baseline report](01-baseline-results.md).

The `0.51` threshold was selected using actual Dart preprocessing on 600
validation images. It accepted 88.5% of that split at 95.10% accepted accuracy;
this is not a guarantee for new field data.

## Limitations and risks

- RealWaste comes from one Australian recovery facility. Geographic, camera,
  background, and lighting domain shift is expected.
- Plastic is the weakest class and is often confused with metal.
- `general_waste` is visually heterogeneous.
- Confidence is validation-specific and abstention cannot eliminate errors.
- Electronics, textiles, vegetation, medical, and hazardous waste are absent
  from the training taxonomy.
- Bin colours follow general Turkish Zero Waste guidance; local rules may vary.
- Physical Android latency/RAM tests and an independent phone field set remain
  outstanding.

## Responsible use

Do not present a prediction as a definitive disposal instruction. For unknown,
multi-object, blurry, or hazardous inputs, direct the user to local guidance.
Monitor errors by class, device, and capture conditions. Review licensing,
representation, and label quality before adding training data.

## Artifact identity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| PyTorch checkpoint | 6,233,709 | `0be7762ff5b94eb070b6bff44b3a5f162e0272106ac1e69c78869e59b4a0a7e1` |
| ONNX FP32 | 6,376,446 | `e1208e1381def0628d43d78a5bfc2a7cfb9d31ea6a31f909925029a3da28900e` |
| LiteRT FP16 | 3,219,264 | `b453d7c57423701e54e4dcdf4a08484716f671ac542f316cb0c156c8dfed2260` |
| LiteRT FP16 Android inline-buffer | 3,218,544 | `f4c93f2d7ded0e69af27e3ac0cd77ed7b8c9501c89e692910b77f8efac61de64` |
| LiteRT INT8 weight-only | 1,832,048 | `dd2e2fdfb16d9864a5f89df8a3eaf6273e5f230f068e8acbbfe77d6f6c275a76` |

Full-INT8 artifacts failed the quality gate and are not deployment candidates.
See [ADR-002](../decisions/ADR-002-int8-deployment-policy.md).

The Android inline-buffer artifact changes only FlatBuffer constant storage;
its logits were verified bit-identical to the source FP16 model.

## Data, licensing, and maintenance

See the [Dataset Card](../data/DATASET_CARD.en.md). Because the RealWaste
license records conflict, the more restrictive CC BY-NC-SA 4.0 terms are
assumed. Public or commercial model-weight distribution needs a separate
review. This is not legal advice.

A changed class order, preprocessing contract, manifest hash, test result, or
artifact hash must be released as a new model version.
