# SmartWaste Vision Mobile

Android için çevrimdışı Flutter istemcisi. Kamera veya galeriden alınan fotoğrafı
uygulama içine gömülü MobileNetV3-Small LiteRT FP16 modeliyle sınıflandırır.

## Çalıştırma

```powershell
flutter pub get
flutter test
flutter run
```

Uygulama en az Android API 26 gerektirir. Model girdisi ve sınıf sırası
`assets/models/model_contract.json` içinde sürümlenmiştir. Mobil ön işleme,
PyTorch/Pillow ile aynı antialias bilinear resize sözleşmesini uygular. Düşük
güven eşiği mobil ön işlenmiş validation setinde `%95` kabul edilen-tahmin
doğruluğu hedefiyle `0.51` olarak kalibre edilmiştir.

Dondurulmuş 598 görsellik test setinde uygulamanın gerçek Dart ön işlemesi ve
LiteRT FP16 model zinciri `%90.47` accuracy ve `%90.56` Macro F1 üretir.

Model altı sınıfı destekler: plastik, cam, metal, kâğıt/karton, organik ve genel
atık. Kutu renkleri Türkiye Sıfır Atık renk skalasına göre gösterilir.
