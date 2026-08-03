# SmartWaste Vision — Benchmark Protokolü

## Amaç

Aynı eğitim checkpoint'inden türetilmiş runtime çıktılarının kalite, boyut,
gecikme ve bellek farklarını ölçmek. Farklı cihaz sonuçları tek bir yarış tablosu
gibi yorumlanmaz; cihaz ve runtime matrisi açıkça raporlanır.

## Model matrisi

| Kimlik | Format | Hassasiyet | Runtime | Ana cihaz |
|---|---|---|---|---|
| `torch_fp32` | `.pth` / state dict | FP32 | PyTorch | Windows CPU |
| `onnx_fp32` | `.onnx` | FP32 | ONNX Runtime CPU | Windows CPU |
| `onnx_int8` | `.onnx` QDQ | INT8 | ONNX Runtime CPU | Windows CPU |
| `tflite_fp16` | `.tflite` | FP16 ağırlık | LiteRT/XNNPACK | Android CPU |
| `tflite_int8` | `.tflite` | Tam INT8 | LiteRT/XNNPACK | Android CPU |

TFLite dosyaları geliştirme bilgisayarında da doğrulanabilir; portföydeki mobil
performans iddiası gerçek Android cihaz ölçümüne dayanır.

## Sabit koşullar

- Batch: 1
- Girdi: RGB, 224×224
- Sınıf sırası ve normalization: model metadata dosyasından
- Test manifesti: bütün formatlarda aynı SHA-256 kayıtları
- CPU thread sayısı: hem 1 thread hem runtime varsayılanı ayrı ölçüm
- Güç modu: cihaz, şarj durumu ve termal durum raporda yazılı
- Ağ: mobil benchmark sırasında kapalı
- Debug build sonuçları portföy benchmark'ı olarak kullanılmaz

## Kalite ölçümü

Her format için:

- Accuracy
- Macro precision, recall ve F1
- Sınıf bazlı precision/recall/F1
- Confusion matrix
- PyTorch referansına top-1 agreement
- Logit veya olasılık farkı özeti
- Güven eşiği uygulanmış coverage ve selective accuracy

INT8 calibration verisi test setine dokunmaz. Quantization kaybı, aynı test
manifestinde PyTorch FP32 sonucuna göre yüzde puan olarak raporlanır.

## Gecikme ölçümü

Her cihaz/runtime çifti için iki ayrı sayı üretilir:

1. **Inference-only:** önceden hazırlanmış tensordan model çıktısına kadar.
2. **Uçtan uca:** dosya decode, orientation, resize/crop, normalize, inference ve
   postprocess dahil.

Protokol:

- Model yükleme süresi ayrı ölçülür.
- En az 50 warm-up çalıştırması yapılır.
- En az 500 zamanlı tekrar yapılır.
- Her örnek için yüksek çözünürlüklü monotonic clock kullanılır.
- Ortalama, medyan, p95, standart sapma, minimum ve maksimum raporlanır.
- Termal throttling belirtisi varsa koşu geçersiz sayılır ve olay kaydedilir.

## Bellek ve boyut

- Dosya boyutu bayt ve MiB olarak kaydedilir.
- Windows'ta model yükleme öncesi ve kararlı inference sırasındaki process RSS
  farkı ölçülür.
- Android'de PSS, uygulama açılışı sonrası baseline ve ısınmış inference sonrası
  `dumpsys meminfo`/uygun profil aracıyla ölçülür.
- Tepe değer ile kararlı durum değeri ayrı tutulur.

## Adil karşılaştırma kuralları

- Bilgisayar ve telefon gecikmeleri doğrudan birbirine bölünerek “x kat hızlı”
  iddiası üretilmez.
- PyTorch ve ONNX yalnız aynı Windows CPU koşusunda doğrudan karşılaştırılır.
- TFLite FP16 ve INT8 yalnız aynı Android cihaz, build ve thread ayarında
  doğrudan karşılaştırılır.
- FP16 CPU üzerinde her zaman hız kazandırmayabilir; boyut kazanımı ayrı
  değerlendirilir.
- INT8'in daha yavaş çıktığı sonuç gizlenmez; runtime/hardware uyumu açıklanır.

## Sonuç tablosu şablonu

| Model | Cihaz | Thread | Acc. | Macro F1 | Boyut MiB | Mean ms | p95 ms | RAM/PSS Δ MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | TBD | 1 | — | — | — | — | — | — |
| ONNX FP32 | TBD | 1 | — | — | — | — | — | — |
| ONNX INT8 | TBD | 1 | — | — | — | — | — | — |
| TFLite FP16 | Android TBD | 1 | — | — | — | — | — | — |
| TFLite INT8 | Android TBD | 1 | — | — | — | — | — | — |

## Tekrarlanabilirlik kaydı

Her koşu şu alanları JSON/CSV olarak saklar: Git commit, model hash, dataset
manifest hash, runtime ve bağımlılık sürümleri, işletim sistemi, CPU/SoC, RAM,
thread/delegate ayarı, warm-up/tekrar sayısı, zaman damgası ve ham ölçümler.
