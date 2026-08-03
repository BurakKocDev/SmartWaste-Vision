# SmartWaste Vision — MVP Uygulama Planı

## Başarı tanımı

MVP aşağıdakilerin tümü gerçekleştiğinde tamamlanmış sayılır:

1. Lisansı ve sınıf eşlemesi belgelenmiş veriyle tek bir MobileNetV3-Small
   modeli eğitilir.
2. Aynı checkpoint'ten PyTorch FP32, ONNX FP32/INT8 ve TFLite FP16/INT8
   çıktıları üretilir.
3. Bütün çalışma biçimleri aynı test manifestiyle doğruluk ve macro F1 açısından
   değerlendirilir.
4. Desteklenen aynı cihazlarda model boyutu, ortalama ve p95 gecikme ile bellek
   kullanımı ölçülür.
5. Flutter Android uygulaması kamera veya galeriden aldığı tek fotoğrafı
   çevrimdışı sınıflandırır; sınıf, güven ve Türkiye renk skalasına uygun kutu
   rengini gösterir.
6. Düşük güvenli veya uygun olmayan girdiler kesin sonuç gibi sunulmaz.
7. Türkçe/İngilizce README, benchmark tablosu, confusion matrix, Model Card,
   Dataset Card, kısa demo videosu ve Docker'lı referans API hazırlanır.

## 2–3 haftalık çalışma sırası

| Gün | Aşama | Çıktı | Durum |
|---:|---|---|---|
| 1 | Dönüşüm riski | Beş formatlı küçük model canary testi | Kısmi geçti; INT8 için QAT gerekli |
| 2–3 | Veri | Lisans kaydı, sınıf haritası, manifest ve split | Tamamlandı; lisans çelişkisi belgeli |
| 4–6 | Model | Transfer learning baseline ve hata analizi | Tamamlandı |
| 7–8 | Optimizasyon | ONNX/TFLite export ve INT8 kalibrasyon | Kısmi; weight-only kabul, full INT8 red |
| 9–10 | Benchmark | CPU ve Android ölçümleri, karşılaştırma tablosu | CPU tamamlandı; Android bekliyor |
| 11–13 | Flutter | Kamera/galeri ve çevrimdışı inference | Modern UI, debug/release-mod APK; kamera, galeri ve uçak modu emülatör smoke testi tamam; fiziksel cihaz bekliyor |
| 14 | API | Docker'lı küçük FastAPI referans servisi | Tamamlandı; health ve gerçek JPEG smoke testi geçti |
| 15 | Portföy | İki dilli belgeler, kartlar, görseller ve video | Belgeler, kartlar, UI v2 görselleri ve final video tamam; gerçek cihaz ölçümü bekliyor |

Takvim iş günü tahminidir. Dönüşüm kapısı başarısızsa kapsam büyütülmez;
uyumsuzluk giderilir veya tek model ailesi içinde daha güvenli export yolu
seçilir.

## Aşamalar ve karar kapıları

### Kapı 0 — Girdi sözleşmesi

- Tek fotoğrafta tek baskın nesne tanımı yazılıdır.
- Çoklu nesne, aşırı bulanıklık ve bilinmeyen nesne davranışı tanımlıdır.
- Android test cihazının model adı, SoC, Android sürümü ve RAM'i kaydedilmiştir.

### Kapı 1 — Dönüşüm

- PyTorch FP32 checkpoint yüklenir.
- ONNX FP32 ve INT8 dosyaları geçerli ve çalıştırılabilirdir.
- TFLite FP16 ve INT8 dosyaları gerçek LiteRT runtime ile çalıştırılabilirdir.
- Sınıf sırası, normalization ve çıktı semantiği beş biçimde aynıdır.
- Dönüşüm test raporu ve kilitli bağımlılık sürümleri vardır.

3 Ağustos 2026 canary sonucu: PyTorch/ONNX FP32 ve LiteRT FP16 geçti. Doğrudan
PTQ ile üretilen ONNX/TFLite INT8 parity kapısını geçmedi; TFLite INT8 ayrıca
varsayılan XNNPACK delegate ile hazırlanamadı. Kapı INT8 açısından açık kalır
ve gerçek baseline sonrasında kısa QAT fine-tune ile tekrar edilir.

### Kapı 2 — Veri

- Her dosyanın kaynak, lisans, hash ve sınıf kaydı manifestte bulunur.
- Benzer veya aynı görüntüler split'ler arasında sızmaz.
- Elektronik sınıfı lisans, kalite ve denge şartını geçmezse MVP altı sınıfa
  indirilir; başka belirsiz web verisi eklenmez.
- Harici telefon test seti eğitim verisinden ayrı tutulur.

### Kapı 3 — Model

- Accuracy, macro F1, sınıf bazlı precision/recall ve confusion matrix vardır.
- Yanlış sınıflandırma galerisi ve confidence kalibrasyonu incelenmiştir.
- Eşik, yalnız validation seti üzerinde seçilmiştir.

### Kapı 4 — Yayın

- Benchmark komutları temiz ortamda yeniden çalışır.
- Test APK'sı ağ olmadan tahmin yapar.
- Gizli anahtar, özel veri veya lisansı belirsiz görüntü depoda yoktur.
- Bilinen sınırlamalar README ve Model Card içinde görünürdür.

## İş bölümü

### Burak

- Hedef Android telefonu ve gerçek cihaz testini sağlamak
- Küçük, bağımsız saha test seti için fotoğraf çekmek
- Şüpheli etiketleri ve yanlış tahminleri doğrulamak
- Demo akışını ve kullanıcı kabul senaryolarını denemek
- Kamuya açık model/veri/APK yayınını onaylamak

### Codex

- Veri manifesti, kalite kontrolleri ve tekrarlanabilir split'i geliştirmek
- Eğitim, değerlendirme, export ve quantization pipeline'ını geliştirmek
- Benchmark araçlarını ve sonuç tablosunu üretmek
- Flutter uygulaması ile Docker'lı FastAPI referansını geliştirmek
- Testleri, Model Card/Dataset Card ve iki dilli belgeleri hazırlamak

### Ortak karar

- Elektronik sınıfının veri kapısını geçip geçmediği
- İlk baseline sonrasında güven eşiği ve hedef metrik revizyonu
- Uygulama ikonları, demo görselleri ve kamuya açık yayın
- MVP'ye yeni bir özellik ekleme isteği
