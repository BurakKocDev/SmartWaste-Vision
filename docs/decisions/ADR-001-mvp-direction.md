# ADR-001 — MVP Ürün ve Teknik Yönü

- Durum: Kabul edildi
- Tarih: 3 Ağustos 2026

## Bağlam

Projenin 2–3 haftada bitebilmesi, VisionStock AI ile tekrar etmemesi ve edge AI
dağıtım becerisini kanıtlaması gerekiyor. En büyük teknik risk PyTorch modelinin
ONNX ve LiteRT/TensorFlow Lite biçimlerine aynı semantikle dönüştürülmesidir.

## Kararlar

1. Ürün, bireysel kullanıcıya genel atık sınıfı yönlendirmesi yapan portföy
   demosudur; akıllı kutu otomasyonu değildir.
2. Girdi tek fotoğrafta tek baskın nesnedir.
3. Hedef platform Android, hedef çalışma biçimi çevrimdışı LiteRT CPU'dur.
4. Masaüstü referansı Windows/x86-64 CPU üzerinde PyTorch ve ONNX Runtime'dır.
5. Model ailesi `torchvision` MobileNetV3-Small'dır. Aynı MVP içinde
   EfficientNet ile model yarışı yapılmaz.
6. Girdi boyutu 224×224, batch boyutu 1'dir.
7. Tek eğitim checkpoint'inden PyTorch FP32, ONNX FP32/INT8 ve TFLite
   FP16/INT8 türetilir.
8. ONNX INT8 için CNN'lere uygun statik quantization ve calibration seti
   kullanılır.
9. TFLite için ilk tercih doğrudan LiteRT Torch dönüşümüdür. Gerekirse yalıtılmış
   ve belgelenmiş bir uyumluluk fallback'i kullanılır; eğitim modeli değişmez.
10. İlk tam eğitimden önce bütün çıktı biçimleri küçük bir canary modelle
    çalıştırılır.
11. Renkler Türkiye Sıfır Atık renk skalasına dayanır ve metin/piktogramla
    birlikte gösterilir.
12. Elektronik sınıfı yalnız lisansı ve kaynağı doğrulanmış yeterli veri varsa
    yer alır; aksi halde MVP altı sınıfla tamamlanır.

## Neden MobileNetV3-Small?

- Mobil/edge kullanım için tasarlanmış küçük bir mimaridir.
- Torchvision ağırlıkları yaklaşık 2,54 milyon parametre ve 9,8 MB dosya
  boyutundadır.
- 224×224 girdide düşük hesaplama maliyeti, CPU benchmark'ında anlamlı bir
  başlangıç sağlar.
- Tek model sınırını korurken FP16/INT8 boyut ve hız etkisini görünür kılar.

İlk baseline kabul hedeflerini karşılamazsa önce veri, sınıf karışıklığı,
kalibrasyon ve eğitim reçetesi incelenir. Mimari değişikliği otomatik değildir ve
yeni ADR gerektirir.

## Sonuçlar

### Olumlu

- VisionStock'tan teknik olarak ayrışan net bir edge AI hikâyesi oluşur.
- Beş runtime çıktısı tek checkpoint'e izlenebilir.
- Uygulama ağ olmadan gerçek cihazda gösterilebilir.
- Benchmark tekrarlanabilir ve CV'de sayısal kanıt sunar.

### Sınırlamalar

- Tek fotoğraf sınıflandırması çoklu atık sahnesini çözmez.
- Veri kaynaklarının sahne farkı yanlış biçimde öğrenilebilir.
- INT8 her cihazda FP16/FP32'den hızlı olmak zorunda değildir.
- Confidence skoru bilinmeyen nesneyi güvenilir biçimde tespit etmez; eşik
  yalnızca kontrollü bir güvenlik katmanıdır.
- Kutu rengi her belediyedeki fiili toplama uygulamasını garanti etmez.

## Dayanaklar

- Torchvision MobileNetV3-Small:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- PyTorch ONNX export:
  https://docs.pytorch.org/docs/stable/onnx.html
- ONNX Runtime quantization:
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- LiteRT Torch:
  https://github.com/google-ai-edge/litert-torch
- Türkiye Sıfır Atık renk skalası:
  https://kastamonu.csb.gov.tr/sifir-atik-renk-skalasi-haber-296215
