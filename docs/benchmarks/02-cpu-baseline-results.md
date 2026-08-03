# CPU Baseline Benchmark Sonucu

## Protokol

- CPU: 11th Gen Intel Core i5-11400H @ 2.70 GHz
- Ortam: WSL2/Ubuntu
- Thread: 4
- Batch: 1
- Girdi: `1×3×224×224`, aynı normalize edilmiş görüntü
- Warmup: 20 inference
- Ölçüm: 100 inference
- Her runtime ayrı process içinde çalıştırıldı.
- Gecikme yalnız model inference çağrısını kapsar; fotoğraf decode ve
  preprocessing dahil değildir.

## Sonuç

| Runtime | Model | Boyut | Ortalama | p95 | Model yüklü RSS | Son RSS | Durum |
|---|---|---:|---:|---:|---:|---:|---|
| PyTorch FP32 | `.pth` | 5,95 MiB | 5,24 ms | 6,17 ms | 308,8 MiB | 325,0 MiB | Kabul |
| ONNX Runtime FP32 | `.onnx` | 6,08 MiB | 1,35 ms | 1,79 ms | 64,8 MiB | 69,5 MiB | Kabul |
| LiteRT FP16 | `.tflite` | 3,07 MiB | 6,80 ms | 8,11 ms | 42,1 MiB | 57,1 MiB | Kabul |
| LiteRT INT8 weight-only | `.tflite` | 1,75 MiB | 0,67 ms | 0,95 ms | 55,9 MiB | 56,0 MiB | Kabul, float compute |
| ONNX full INT8 PTQ | `.onnx` | 1,72 MiB | 3,76 ms | 4,57 ms | 59,2 MiB | 64,8 MiB | Accuracy nedeniyle red |
| LiteRT full INT8 PTQ | `.tflite` | 1,84 MiB | 7,03 ms | 8,04 ms | 42,5 MiB | 45,0 MiB | Accuracy/XNNPACK nedeniyle red |

RSS, runtime bağımlılıklarını içeren süreç belleğidir; yalnız tensor arena
ölçümü değildir. “Model yüklü RSS” inference öncesi, “Son RSS” warmup ve ölçüm
sonrasındaki anlık süreç belleğidir.

## Yorum

- Bu CPU'da ONNX FP32, PyTorch FP32'den yaklaşık 3,9 kat daha hızlıdır.
- LiteRT INT8 weight-only en küçük ve en hızlı kabul edilen artifact oldu;
  buna rağmen tam-integer compute iddiası yapılamaz.
- FP16 depolama boyutunu yaklaşık yarıya indirir ancak x86 CPU'da hız avantajı
  sağlamaz. Asıl hedefi Android cihaz benchmarkıdır.
- Accuracy kapısını geçmeyen full INT8 modellerin küçük veya hızlı olması onları
  deploy edilebilir yapmaz.

Ham rapor yerel `artifacts/benchmarks/cpu_baseline.json` dosyasındadır.
