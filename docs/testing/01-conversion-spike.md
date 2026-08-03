# SmartWaste Vision — İlk Gün Dönüşüm Testi

## Amaç

Tam veri hazırlığı ve eğitime yatırım yapmadan önce MobileNetV3-Small modelinin
PyTorch'tan ONNX ve TFLite/LiteRT biçimlerine dönüştürülebildiğini, quantization
uyumluluğunu ve aynı preprocessing/çıktı semantiğini kanıtlamak.

Bu test proje takvimindeki ilk teknik kapıdır.

## Canary model

- `torchvision.models.mobilenet_v3_small`
- ImageNet varsayılan ağırlıkları
- Son sınıflandırıcı hedef sınıf sayısına değiştirilmiş
- `eval()` modu
- Girdi: `float32`, NCHW, `[1, 3, 224, 224]`
- Sabit seed ve sabit on örnek görüntü

Canary eğitimin doğruluğunu kanıtlamaz. Yalnız graph, operator, metadata,
quantization ve runtime uyumluluğunu sınar.

## Üretilecek çıktılar

1. `canary.pth` — PyTorch FP32 referans
2. `canary_fp32.onnx` — ONNX FP32
3. `canary_int8.onnx` — ONNX Runtime statik INT8 QDQ
4. `canary_fp16.tflite` — TFLite FP16
5. `canary_int8.tflite` — TFLite tam INT8
6. `labels.json` ve `preprocess.json` — tek sınıf sırası ve girdi sözleşmesi
7. `conversion_report.json` — sürümler, hash'ler, parity ve hata kayıtları

## Uygulama sırası

### 1. Ortamı kilitle

- Dönüşüm için ayrı Python ortamı oluştur.
- Python, PyTorch, torchvision, ONNX, ONNX Runtime ve LiteRT Torch sürümlerini
  kilitle.
- Dönüşüm aracı Windows'ta desteklenmezse belgelenmiş Linux/Docker/WSL ortamı
  kullan; eğitim ortamıyla karıştırma.

### 2. PyTorch referansını dondur

- Modeli yükle ve checkpoint hash'ini kaydet.
- Normalization'ı model metadata'sından al.
- On sabit girdinin logits, softmax ve top-1 çıktısını kaydet.

### 3. ONNX FP32

- Güncel `torch.export` tabanlı `torch.onnx.export` yolunu kullan.
- Sabit batch 1 ve 224×224 şekliyle başla; gereksiz dynamic axis ekleme.
- `onnx.checker` ve ONNX Runtime CPU ile çalıştır.
- PyTorch ile logits farkını ve top-1 agreement'ı raporla.

### 4. ONNX INT8

- ONNX Runtime preprocessing/shape inference adımını ayrı çalıştır.
- CNN için statik quantization ve train split'inden temsilî calibration girdisi
  kullan.
- İlk tercih S8S8 QDQ'dur.
- Boyut, operator kapsaması, parity ve olası accuracy kaybını kaydet.

### 5. TFLite FP16 ve INT8

- İlk tercih doğrudan `litert_torch.convert` yoludur.
- FP16 ve tam INT8'in gerçekten beklenen tensor tiplerini içerdiğini model
  analyzer/interpreter ile doğrula; dosya adını kanıt sayma.
- INT8 için sınıf dengeli temsilî calibration girdisi kullan.
- LiteRT interpreter ile aynı on örneği çalıştır.
- Flutter entegrasyonundan önce masaüstü runtime parity'sini tamamla.

### 6. Gerçek Android smoke testi

- Release/profile build içinde her iki TFLite modelini yükle.
- Bir sabit görselde top-1, confidence, model yükleme ve tek inference süresini
  kaydet.
- Uçak modunda aynı sonucu doğrula.

## Geçiş ölçütleri

- Beş model dosyası hedef runtime tarafından açılır.
- Her dosyanın SHA-256 hash'i ve byte boyutu kaydedilir.
- FP32 export'lar PyTorch ile en az %99 top-1 agreement gösterir.
- INT8 export'lar PyTorch ile en az %97 top-1 agreement gösterir.
- NaN/Inf çıktı yoktur.
- Sınıf sırası ve preprocessing beş çalışma biçiminde aynıdır.
- TFLite INT8 modelinin giriş/çıkış dtype ve scale/zero-point bilgisi rapordadır.
- Gerçek Android cihazda çevrimdışı smoke test geçer.

## Başarısızlık halinde sıra

1. Modelin `eval()` durumu, input layout, normalization ve classifier head'i
   kontrol edilir.
2. Desteklenmeyen operator ve export raporu incelenir.
3. Sabit input shape ve basitleştirilmiş wrapper denenir.
4. Quantization calibration örnekleri ve tensor tipleri doğrulanır.
5. LiteRT Torch sürüm uyumu ayrı ortamda yeniden üretilir.
6. Gerekirse ONNX/TensorFlow uyumluluk fallback'i izole edilir ve farklar
   belgelenir.

Bu sıra tamamlanmadan mimari değiştirilmez, ikinci model eklenmez ve tam eğitim
başlatılmaz.
