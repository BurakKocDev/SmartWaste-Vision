# SmartWaste Vision — Dönüşüm Canary Sonucu

## Karar

3 Ağustos 2026 tarihli ilk dönüşüm kapısı **kısmi geçti**.

- PyTorch FP32 checkpoint üretildi.
- ONNX FP32, PyTorch ile `%100` top-1 agreement gösterdi.
- LiteRT FP32 ara modeli ve gerçek FP16 ağırlıklı LiteRT modeli `%100` top-1
  agreement gösterdi.
- ONNX statik INT8 QDQ ve LiteRT tam INT8 dosyaları üretildi ancak parity
  eşiğini geçmedi.
- LiteRT INT8 dosyası reference kernel ile çalıştı; varsayılan XNNPACK delegate
  modeli hazırlayamadı.

Bu nedenle FP32/FP16 zinciri kabul edildi. INT8 dosyaları tanı artifactıdır ve
uygulamaya eklenmeyecektir. Gerçek FP32 baseline eğitiminden sonra kısa bir QAT
fine-tune, ONNX ve LiteRT INT8 için zorunlu optimizasyon kapısıdır.

## Sabit sözleşme

- Model: `MobileNetV3-Small`
- Çıkış: 6 logits
- Girdi: `float32`, NCHW, `[1, 3, 224, 224]`
- Validation: sınıf başına 3, toplam 18 sabit RealWaste görüntüsü
- Calibration: sınıf başına 50, toplam 300 sabit RealWaste görüntüsü
- Canary head: ilgili ImageNet satırlarıyla başlatıldı; hedef veriyle
  eğitilmediği için bu test accuracy ölçümü değildir.

## Parity sonucu

| Çalışma biçimi | Top-1 agreement | Ortalama mutlak logits farkı | En büyük fark | Runtime | Karar |
|---|---:|---:|---:|---|---|
| ONNX FP32 | %100 | 0,000005 | 0,000021 | ONNX Runtime CPU | Geçti |
| LiteRT FP32 ara model | %100 | 0,000010 | 0,000037 | XNNPACK | Geçti |
| LiteRT FP16 | %100 | 0,014449 | 0,059378 | XNNPACK | Geçti |
| ONNX INT8 QDQ | %27,8 | 2,789691 | 8,928818 | ONNX Runtime CPU | Kaldı |
| LiteRT tam INT8 | %27,8 | 3,332292 | 8,423105 | Reference kernel | Kaldı |

Geçiş eşikleri FP32/FP16 için `%99`, INT8 için `%97` top-1 agreement idi.
Bütün çıktılar sonlu kaldı; NaN/Inf görülmedi.

## Artifact boyutları

| Artifact | Boyut |
|---|---:|
| `canary.pth` | 5,94 MiB |
| `canary_fp32.onnx` | 6,08 MiB |
| `canary_int8.onnx` | 1,72 MiB |
| `canary_fp16.tflite` | 3,07 MiB |
| `canary_int8.tflite` | 1,84 MiB |

SHA-256 değerleri ve tam byte boyutları yerel
`artifacts/conversion_canary/conversion_report.json` raporundadır.

## INT8 incelemesi

Aşağıdaki seçenekler aynı sabit girdilerle ayrıca denendi:

- S8/S8 ve U8/S8 QDQ
- U8/U8 QDQ ve QOperator
- per-channel ve per-tensor ağırlık quantization
- reduced range
- erken hassas MobileNet bloklarını FP32 bırakma
- 30 ve 300 görüntülük calibration setleri
- PyTorch PT2E observer → LiteRT/ONNX export fallback'i

Hiçbiri `%97` parity kapısını geçmedi. PT2E graph her iki biçime export edildi
ve LiteRT tarafında XNNPACK ile açıldı, fakat PTQ kalite kaybı devam etti. Eşiği
düşürmek yerine QAT kararı alındı.

## Kilitli ortam

- PyTorch `2.12.1+cpu`
- torchvision `0.27.1+cpu`
- ONNX `1.21.0`
- ONNX Runtime `1.28.0`
- LiteRT Torch `0.9.2`
- AI Edge Quantizer `0.8.0`
- AI Edge LiteRT `2.1.6`

Koşu WSL2/Ubuntu üzerinde `requirements-conversion.lock` ile yapıldı.

## Sıradaki teknik adım

1. Altı sınıflı, group-aware `%70/%15/%15` veri splitini dondur.
2. FP32 transfer-learning baseline'ını eğit ve validation metriklerini kaydet.
3. Başarılı checkpoint'i FP16 olarak export et.
4. Aynı checkpoint üzerinde kısa QAT fine-tune uygula.
5. ONNX/TFLite INT8 parity, accuracy ve XNNPACK/Android smoke testini yeniden
   çalıştır.
