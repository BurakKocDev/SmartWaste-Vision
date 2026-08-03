# SmartWaste Vision

[English](README.en.md) · Türkçe

SmartWaste Vision, tek bir atık nesnesinin fotoğrafını sınıflandıran ve aynı
eğitilmiş modelin PyTorch, ONNX ve LiteRT/TensorFlow Lite çalışma biçimlerindeki
doğruluk, gecikme, bellek ve model boyutu farklarını karşılaştıran çevrimdışı
öncelikli bir edge AI portföy projesidir.

> Portföy iddiası: Aynı görüntü sınıflandırma modelini sunucu, masaüstü CPU ve
> Android edge ortamlarına dağıtıp doğruluk, gecikme, bellek ve boyut açısından
> karşılaştırdım.

## Güncel durum

- Tarih: 4 Ağustos 2026
- Aşama: Aşama 5 — gerçek Android cihaz doğrulaması bekleniyor
- Kod durumu: Veri, eğitim, export, evaluation, CPU benchmark, Flutter ve Docker API çalışıyor
- Veri durumu: 4.752/4.752 görsel sağlam; kaynak lisans metinleri çelişkili
- Model sonucu: test Accuracy `%90,80`, Macro F1 `0,9093`
- Mobil zincir: Dart ön işleme + LiteRT FP16 test Accuracy `%90,47`, Macro F1 `0,9056`
- Dönüşüm durumu: ONNX FP32/LiteRT FP16 geçti; LiteRT INT8 weight-only kabul edildi
- Uygulama: Modern eco-tech arayüz, kamera/galeri, kutu rengi, `%51` düşük güven eşiği ve offline tahmin hazır; debug ve release-mod APK emülatör smoke testleri geçti
- API: Docker imajı build edildi; health ve gerçek JPEG sınıflandırma smoke testi geçti
- Portföy: İki dilli README, Model/Dataset Card, confusion matrix, mobil arayüz görselleri ve final demo videosu hazır
- Takvim: Android gerçek cihaz benchmarkıyla MVP kapanacak

## Mobil arayüz

| Ana ekran | Tahmin sonucu |
|---|---|
| ![SmartWaste Vision ana ekran](docs/assets/mobile-home-v2.png) | ![SmartWaste Vision tahmin sonucu](docs/assets/mobile-result-v2.png) |

Arayüz; belirgin bir offline durumu, fotoğraf çekim alanı, sınıf ve kutu
yönlendirmesi, ilk üç olasılık ve düşük güvenli girdiler için ayrı bir
uyarı deneyimi sunar.

[Final demo videosunu izle](docs/assets/smartwaste-vision-demo-tr.mp4)

## VisionStock AI'dan farkı

| Boyut | SmartWaste Vision | VisionStock AI |
|---|---|---|
| CV görevi | Tek nesne görüntü sınıflandırma | Object detection ve sayım |
| Çalışma yeri | Android cihazda çevrimdışı | FastAPI üzerinde sunucu tarafı |
| Ana teknik iddia | Export, quantization ve edge benchmark | Raf analizi ve stok iş akışı |
| Veri | Genel atık malzeme türleri | Altı belirli içecek SKU'su |
| Ürün kapsamı | Küçük tüketici yönlendirme demosu | B2B raf kontrol pilotu |

İki proje Flutter ve görüntü girdisini paylaşsa da model görevi, dağıtım biçimi
ve portföy iddiası farklıdır.

## Onaylanan MVP yönü

- Tek fotoğrafta tek baskın atık nesnesi
- Altı sınıf: plastik, cam, metal, kâğıt/karton, organik ve genel atık
- Tek model ailesi: ImageNet ağırlıklı `MobileNetV3-Small`
- PyTorch ile transfer learning
- PyTorch FP32
- ONNX Runtime FP32; full INT8 yalnız kalite kapısı deneyi
- LiteRT/TensorFlow Lite FP16 ve INT8 weight-only; full INT8 yalnız red deneyi
- Android öncelikli küçük Flutter uygulaması
- Kamera veya galeri, sınıf, güven skoru, kutu rengi ve düşük güven uyarısı
- Tamamen çevrimdışı mobil tahmin
- Docker içinde küçük PyTorch/FastAPI referans API'si

## Kapsam dışı

- Nesne tespiti ve semantic segmentation
- Aynı fotoğrafta birden fazla atığı ayırma
- Canlı video veya geri dönüşüm bandı
- Harita ve geri dönüşüm noktaları
- Puan, rozet, hesap ve kapsamlı yönetim paneli
- Belediyeye göre kesin bertaraf kararı
- Tehlikeli atıklar için profesyonel karar

## Proje belgeleri

- [MVP kapsamı](docs/business-analysis/01-mvp-scope.md)
- [MVP teknik kararı](docs/decisions/ADR-001-mvp-direction.md)
- [Veri ve sınıf planı](docs/data/01-dataset-and-taxonomy-plan.md)
- [Altı sınıflı split sonucu](docs/data/02-split-results.md)
- [Benchmark protokolü](docs/benchmarks/01-benchmark-plan.md)
- [CPU benchmark sonucu](docs/benchmarks/02-cpu-baseline-results.md)
- [İlk gün dönüşüm testi](docs/testing/01-conversion-spike.md)
- [Dönüşüm canary sonucu](docs/testing/02-conversion-spike-results.md)
- [FP32 baseline sonucu](docs/model/01-baseline-results.md)
- [Model Card — Türkçe](docs/model/MODEL_CARD.md) / [English](docs/model/MODEL_CARD.en.md)
- [Dataset Card — Türkçe](docs/data/DATASET_CARD.md) / [English](docs/data/DATASET_CARD.en.md)
- [INT8 deployment politikası](docs/decisions/ADR-002-int8-deployment-policy.md)
- [Flutter MVP ve mobil ön işleme sonucu](docs/mobile/01-flutter-mvp.md)
- [Docker/PyTorch API](api/README.md)
- [Portföy yayın kontrol listesi](docs/release/01-portfolio-release-checklist.md)
- [Uygulama planı](PROJECT_PLAN.md)

## Sonuç tablosu

| Runtime | Test Accuracy | Macro F1 | Boyut | Ortalama / p95 CPU |
|---|---:|---:|---:|---:|
| PyTorch FP32 | %90,80 | 0,9093 | 5,95 MiB | 5,24 / 6,17 ms |
| ONNX FP32 | %90,80 | 0,9093 | 6,08 MiB | 1,35 / 1,79 ms |
| LiteRT FP16 | %90,64 | 0,9074 | 3,07 MiB | 6,80 / 8,11 ms |
| LiteRT INT8 weight-only | %89,80 | 0,8989 | 1,75 MiB | 0,67 / 0,95 ms |

CPU gecikmesi WSL2 üzerindeki Intel i5-11400H'de, batch 1 ve dört thread ile,
decode/ön işleme hariç ölçüldü. Mobil uygulamanın gerçek Dart ön işlemesiyle
FP16 test sonucu `%90,47` Accuracy / `0,9056` Macro F1'dır. Full INT8 sonuçları
kalite kapısını geçmedi ve deploy edilmedi. Ayrıntılı protokol
[CPU benchmark raporundadır](docs/benchmarks/02-cpu-baseline-results.md).

## Hızlı başlangıç

Flutter testleri, debug ve release-mod test APK'ları:

```powershell
cd apps/mobile
flutter pub get
flutter test
cd ../..
.\scripts\build_android.ps1 -Configuration Debug
.\scripts\build_android.ps1 -Configuration Release
```

Release-mod test APK `artifacts/mobile/smartwaste-vision-release.apk` konumuna
yazılır. Bu APK şu anda test amaçlı debug anahtarıyla imzalanır; Play Store
dağıtım paketi değildir.

Docker referans API:

```bash
docker build -t smartwaste-vision-api .
docker run --rm -p 8000:8000 smartwaste-vision-api
curl -F "file=@waste.jpg;type=image/jpeg" http://localhost:8000/v1/classify
```

Uygulama tahmin için ağa bağlanmaz; API aynı checkpoint'in sunucu dağıtımını
göstermek için ayrı bir referanstır.

## Geçici portföy sırası

1. SmartWaste Vision
2. VisionStock AI — fotoğraflar hazır olunca
3. ModelLens
4. FraudStream
5. DevGuard AI

## Sıradaki iş

Sırada release-mod APK'yı gerçek Android telefonda çalıştırmak; kamera/galeri
ve uçak modunda tahmini doğrulamak, model gecikmesi, uçtan uca gecikme, p95 ve
RAM değerlerini kaydetmek var. Final demo videosu yenilenen arayüzle hazırlandı.
Accuracy kapısını geçmeyen full INT8 artifactları uygulamaya alınmayacaktır.
