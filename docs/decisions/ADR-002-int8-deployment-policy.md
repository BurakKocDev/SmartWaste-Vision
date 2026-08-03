# ADR-002 — INT8 Deployment Politikası

- Durum: Kabul edildi
- Tarih: 3 Ağustos 2026

## Bağlam

MobileNetV3-Small için ONNX Runtime statik INT8 QDQ ve LiteRT tam INT8 PTQ
artifactları başarıyla üretildi. Ancak hem canary hem eğitimli checkpoint
üzerinde parity ve test accuracy ciddi biçimde düştü. LiteRT tam INT8 modeli
ayrıca varsayılan XNNPACK delegate ile hazırlanamadı.

S8/S8, U8/S8, U8/U8, QDQ, QOperator, per-channel/per-tensor, reduced range,
seçici erken katman quantization, 30/300 calibration örneği ve üç epoch PT2E QAT
denendi. Hiçbiri kalite kapısını geçmedi.

## Karar

MVP'de:

- PyTorch FP32 referans olarak tutulur.
- ONNX FP32 sunucu/CPU deployment adayıdır.
- LiteRT FP16 ana mobil deployment adayıdır.
- LiteRT INT8 weight-only ikinci mobil/edge adayıdır.
- ONNX full INT8 ve LiteRT full INT8 yalnız başarısız deney artifactı ve
  benchmark satırı olarak tutulur; uygulamaya eklenmez.

INT8 weight-only model “tam INT8” diye adlandırılmaz. Ağırlık depolaması int8,
aktivasyon ve compute float'tır.

## Sonuçlar

- Mobil uygulama ilk entegrasyonda FP16 ile başlar.
- Weight-only INT8 model, gerçek Android cihazda accuracy ve latency smoke
  testini geçerse kullanıcıya seçilen production artifact olabilir.
- Full INT8 yeniden çalışma ancak LiteRT Torch/AI Edge Quantizer sürümü veya
  model ailesi kontrollü bir deneyle değiştiğinde açılır.
- MVP kapsamı ikinci bir model ailesi eklenerek büyütülmez.
