# Model Card — SmartWaste Vision MobileNetV3-Small

[English version](MODEL_CARD.en.md)

## Model özeti

SmartWaste Vision, tek fotoğraftaki tek baskın atık nesnesini altı malzeme
sınıfından birine atayan görüntü sınıflandırma modelidir. ImageNet ağırlıklı
`MobileNetV3-Small`, PyTorch ile RealWaste üzerinde transfer learning yapılarak
eğitilmiştir.

- Sürüm: MVP baseline, 3 Ağustos 2026
- Sınıflar: `plastic`, `glass`, `metal`, `paper_cardboard`, `organic`,
  `general_waste`
- Eğitim: 2.800 train / 600 validation; test 598 görüntüyle donduruldu
- Girdi: RGB, kısa kenar 256, merkez crop `224×224`, ImageNet mean/std
- Çıktı: altı logit; kullanıcıya softmax top-1 güveni gösterilir
- Mobil belirsizlik eşiği: `0,51`; altındaki sonuç “Emin değilim” olarak sunulur

## Kullanım amacı

Model; eğitim, portföy ve genel atık ayrıştırma farkındalığı için, tek ve belirgin
bir nesnenin plastik/cam/metal/kâğıt-karton/organik/genel atık sınıfını tahmin
etmek üzere tasarlanmıştır. Flutter uygulamasında tamamen çevrimdışı; referans
API'de PyTorch FP32 olarak çalışır.

Model şu işler için tasarlanmamıştır:

- aynı karede birden fazla nesneyi bulmak veya ayırmak,
- tehlikeli, tıbbi ya da elektronik atık hakkında güvenlik kararı vermek,
- belediyeye özgü bağlayıcı bertaraf talimatı üretmek,
- endüstriyel bantta canlı video sınıflandırmak,
- insan denetimi olmadan operasyonel veya ticari karar vermek.

## Performans

Model seçimi yalnız validation Macro F1 ile yapıldı; test split'i seçimde
kullanılmadı.

| Runtime | Test Accuracy | Macro F1 | Durum |
|---|---:|---:|---|
| PyTorch FP32 | %90,80 | 0,9093 | Referans |
| ONNX FP32 | %90,80 | 0,9093 | Kabul |
| LiteRT FP16, Python/Pillow ön işleme | %90,64 | 0,9074 | Kabul |
| LiteRT FP16, gerçek Dart ön işleme | %90,47 | 0,9056 | Flutter modeli |
| LiteRT INT8 weight-only | %89,80 | 0,8989 | Kabul; compute float |
| ONNX full INT8 PTQ | %24,08 | 0,0823 | Red |
| LiteRT full INT8 PTQ | %10,54 | 0,0318 | Red |

PyTorch FP32 testinde sınıf F1 değerleri: plastik `0,843`, cam `0,930`, metal
`0,902`, kâğıt/karton `0,959`, organik `0,929`, genel atık `0,894`.
[Confusion matrix](../assets/confusion_matrix.svg) ve ayrıntılı
[baseline raporu](01-baseline-results.md) depodadır.

Mobil eşik gerçek Dart ön işlemesiyle 600 validation görselinde seçildi. `0,51`
eşiği validation örneklerinin `%88,5`'ini kabul eder; bu alt kümedeki doğruluk
`%95,10`'dur. Bu oran yeni saha verisinde garanti değildir.

## Bilinen sınırlamalar ve riskler

- Eğitim verisi tek bir geri kazanım tesisindeki Avustralya atıklarından gelir;
  farklı ülkeler, arka planlar, telefon kameraları ve ışık koşullarında domain
  shift beklenir.
- Plastik en zayıf sınıftır ve özellikle metal ile karışır.
- `general_waste` görsel olarak heterojendir.
- Confidence kalibrasyonu validation split'ine özeldir; düşük güven uyarısı
  yanlış sonuçları tamamen engellemez.
- Eğitim sınıflarında elektronik, tekstil, bitki, tıbbi ve tehlikeli atık yoktur.
- Kutu rengi genel Türkiye Sıfır Atık skalasını izler; yerel toplama kuralları
  farklı olabilir.
- Gerçek Android cihaz latency/RAM ölçümü ve bağımsız telefon saha seti henüz
  tamamlanmamıştır.

## Sorumlu kullanım

Tahmini kesin bertaraf talimatı gibi sunmayın. Bilinmeyen, çoklu, bulanık veya
tehlikeli girdilerde kullanıcıdan yerel kuralları kontrol etmesini isteyin.
Hataları sınıf, cihaz ve çekim koşuluna göre izleyin; yeniden eğitim öncesi yeni
verinin lisansını, temsiliyetini ve etiket kalitesini inceleyin.

## Artifact kimliği

| Artifact | Bayt | SHA-256 |
|---|---:|---|
| PyTorch checkpoint | 6.233.709 | `0be7762ff5b94eb070b6bff44b3a5f162e0272106ac1e69c78869e59b4a0a7e1` |
| ONNX FP32 | 6.376.446 | `e1208e1381def0628d43d78a5bfc2a7cfb9d31ea6a31f909925029a3da28900e` |
| LiteRT FP16 | 3.219.264 | `b453d7c57423701e54e4dcdf4a08484716f671ac542f316cb0c156c8dfed2260` |
| LiteRT FP16 Android inline-buffer | 3.218.544 | `f4c93f2d7ded0e69af27e3ac0cd77ed7b8c9501c89e692910b77f8efac61de64` |
| LiteRT INT8 weight-only | 1.832.048 | `dd2e2fdfb16d9864a5f89df8a3eaf6273e5f230f068e8acbbfe77d6f6c275a76` |

Tam INT8 artifactları kalite kapısını geçmediği için dağıtıma uygun değildir.
Quantization kararı [ADR-002](../decisions/ADR-002-int8-deployment-policy.md)
içinde belgelenmiştir.

Android inline-buffer artifactı yalnız FlatBuffer sabitlerinin depolama biçimini
değiştirir; kaynak FP16 modelle logit farkı `0,0` olarak doğrulanmıştır.

## Veri, lisans ve bakım

Veri ayrıntıları [Dataset Card](../data/DATASET_CARD.md) içindedir. RealWaste
lisans kayıtları çeliştiği için daha kısıtlayıcı CC BY-NC-SA 4.0 varsayılır;
model ağırlıklarının kamuya açık veya ticari dağıtımı ayrıca incelenmelidir.
Bu belge hukuki görüş değildir.

Yeni model sürümü; sınıf sırası, preprocessing, manifest hash'i, test sonuçları
ve artifact hash'leri değiştiğinde yeni bir sürüm olarak yayınlanmalıdır.
