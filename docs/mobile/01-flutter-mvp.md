# Flutter MVP ve mobil ön işleme sonucu

## Uygulama kapsamı

Android öncelikli Flutter istemcisi `apps/mobile` altında oluşturuldu. İlk sürüm:

- kameradan veya galeriden fotoğraf alır,
- gömülü LiteRT FP16 modelini tamamen cihaz içinde çalıştırır,
- atık sınıfını ve softmax güven skorunu gösterir,
- Türkiye Sıfır Atık renk skalasına göre kutu rengini gösterir,
- düşük güvende sonucu kesin hüküm gibi sunmayıp “Emin değilim” uyarısı verir,
- fotoğrafı bir API'ye veya sunucuya göndermez.

Desteklenen sınıflar `plastic`, `glass`, `metal`, `paper_cardboard`, `organic`
ve `general_waste` sınıflarıdır. Kutu renkleri sırasıyla sarı, yeşil, açık gri,
mavi, kahverengi ve koyu gridir. Renk eşlemesinin kaynağı Çevre, Şehircilik ve
İklim Değişikliği Bakanlığı'nın yayımladığı [Sıfır Atık renk
skalasıdır](https://kastamonu.csb.gov.tr/sifir-atik-renk-skalasi-haber-296215).

## Arayüz v2

Mobil arayüz, MVP işlevlerini büyütmeden modern bir eco-tech görsel dil ile
yenilendi. Yeni sürümde:

- çevrimdışı çalışma ve gizlilik ilk bakışta görünür,
- kamera ve galeri eylemleri tek bir odak kartında sunulur,
- yüksek güvenli sonuçta sınıf, kutu rengi, süre ve ilk üç olasılık gösterilir,
- `%51` altı sonuçlar ayrı sarı uyarı kartı ve yeniden çekim önerisiyle sunulur,
- küçük ekranlarda kaydırılabilir düzen korunur.

| Ana ekran | Yüksek güvenli sonuç |
|---|---|
| ![Mobil ana ekran](../assets/mobile-home-v2.png) | ![Mobil tahmin sonucu](../assets/mobile-result-v2.png) |

## Model sözleşmesi

Uygulamaya `smartwaste_fp16.tflite` gömülür. Sınıf sırası, tensör biçimi,
normalizasyon, resize/crop adımları ve confidence eşiği
`assets/models/model_contract.json` dosyasında sürümlenir.

- Girdi: float32 `1×3×224×224`, NCHW
- Resize: kısa kenar 256, Pillow uyumlu antialias bilinear
- Crop: merkezden `224×224`
- Normalizasyon: ImageNet mean/std
- Çıktı: altı float32 logit
- Düşük güven eşiği: `0,51`

Dart `image` paketinin standart linear resize'ı validation Accuracy değerini
`%86,50`'ye düşürdüğü için kullanılmadı. Pillow'ın bilinear antialias katsayı
hesabı Dart'ta ayrık iki geçişli olarak uygulandı. Bu sürüm aynı validation
setinde `%91,17` Accuracy değerini geri kazandı.

## Mobil ön işleme doğrulaması

Confidence eşiği yalnızca 600 görsellik validation split'inde seçildi. `0,51`
eşikte görsellerin `%88,50`'si kabul edilir; kabul edilen tahminlerin doğruluğu
`%95,10`'dur.

Dondurulmuş test split'inde gerçek Dart ön işlemesiyle üretilen tensörler ve
LiteRT FP16 model sonucu:

| Ölçüt | Sonuç |
|---|---:|
| Görsel | 598 |
| Accuracy | %90,47 |
| Macro F1 | 0,9056 |

Karşılaştırma için Python/Pillow ön işlemeli LiteRT FP16 sonucu `%90,64`
Accuracy ve `0,9074` Macro F1'dır. Mobil ön işlemenin farkı test setinde yalnızca
bir ek top-1 hataya karşılık gelir.

## Android build

- Flutter: 3.32.5
- Dart: 3.8.1
- compile SDK: 36
- target SDK: 35
- minimum SDK: 26
- NDK: 27.0.12077973

Windows çalışma dizinindeki Türkçe karakter nedeniyle Gradle çıktısı ASCII
bir geçici dizine yönlendirilir. Windows AOT snapshotter release sırasında kaynak
yolunu da okuduğu için release build temiz, ASCII-only bir geçici proje
kopyasından çalışır. Tekrarlanabilir build komutları:

```powershell
.\scripts\build_android.ps1 -Configuration Debug
.\scripts\build_android.ps1 -Configuration Release
```

Script APK'ları `artifacts/mobile` altına kopyalar ve SHA-256 değerini yazdırır.

Doğrulanan debug APK:

- Boyut: 204.532.721 bayt (`195,06 MiB`)
- SHA-256: `4d48da81f37db28a698ff03c6026f652b5c218dd46c2eb2ebfa1b8bd985dc9fe`
- Paket içindeki Android uyumlu FP16 model SHA-256:
  `f4c93f2d7ded0e69af27e3ac0cd77ed7b8c9501c89e692910b77f8efac61de64`

Bu dosya geliştirme amaçlı, tüm debug sembollerini ve Android ABI'larını içeren
bir APK'dır; yayın boyutu olarak yorumlanmamalıdır.

Doğrulanan release-mod test APK:

- Boyut: 79.417.714 bayt (`75,74 MiB`)
- SHA-256: `8fe0e208023c56f6eb96eac88e59c4ecb103a49f4925b7adf8db879d3f09388b`
- Paket içindeki model/contract SHA-256 eşleşmesi: geçti

Bu APK AOT-derlenmiş release-mod uygulamadır ancak test kolaylığı için debug
anahtarıyla imzalanır. Play Store'a yüklenecek production-signed AAB değildir.

## Açık doğrulama

APK derlenmiş ve içindeki model hash'i Android uyumlu FP16 artifact ile
eşleşmiştir. Android 16/API 36 x86_64 emülatörde uygulama açılışı, galeri seçimi,
offline inference, düşük güven uyarısı ve kutu rengi smoke testi geçmiştir.

Emülatör uçak modunda (`airplane_mode_on=1`, Wi-Fi kapalı) aşağıdaki işlev
kontrolleri geçti:

| Girdi | Akış | Sonuç | Güven | Model süresi | UI davranışı |
|---|---|---|---:|---:|---|
| `Glass_99.jpg` | Galeri | Cam / Yeşil | %94,9 | 40 ms | Normal sonuç |
| `Cardboard_127.jpg` | Galeri | Kâğıt/karton / Mavi | %86,7 | 20 ms | Normal sonuç |
| `Plastic_10.jpg` | Galeri | Plastik / Sarı | %46,7 | 21 ms | “Emin değilim” |
| Emülatör sanal sahnesi | Kamera | Kâğıt/karton / Mavi | %62,4 | 52 ms | Normal sonuç |

`Plastic_10.jpg` güveni `%51` eşiğinin altında olduğu için düşük güven davranışı
doğru çalıştı. Kamera izni, çekim, fotoğrafın uygulamaya dönüşü ve sonrasında
inference da tamamlandı. İlk cold inference `124 ms`, sonraki warm inference
değerleri `20–52 ms` aralığında gözlendi.

Tahmin sonrasında tek anlık `dumpsys meminfo` ölçümü `262.611 KiB TOTAL PSS` ve
`347.004 KiB TOTAL RSS` gösterdi. Bu değerlerin tamamı x86_64 debug emülatör
smoke gözlemidir; fiziksel telefon benchmarkı değildir.

Release-mod APK aynı Android 16/API 36 x86_64 emülatöre kurulup galeri girdisiyle
AOT inference tamamladı. Tahmin sonrası tek anlık ölçüm `120.805 KiB TOTAL PSS`
ve `204.520 KiB TOTAL RSS` gösterdi. Bu, debug/release davranışını ayırt etmek
için bir smoke gözlemidir; telefon RAM benchmarkı olarak raporlanmaz.

Gerçek Android cihaz açılış, kamera/galeri, model latency, uçtan uca latency,
p95 ve RAM ölçümleri sıradaki kapıdır. x86 WSL ve emülatör sonuçları Android
telefon performansı diye sunulmaz.

## Android LiteRT buffer uyumluluğu

İlk emülatör smoke testinde LiteRT 2.1.6 ile üretilen FP16 modelin sabit
ağırlıkları yeni offset-backed buffer alanlarında tuttuğu görüldü. Flutter
paketindeki Android runtime modeli açsa da inference sırasında `Input tensor 69
lacks data` hatası verdi.

60 sabit buffer, veri baytları değiştirilmeden klasik inline TFLite buffer
biçimine yeniden paketlendi. Kaynak ve Android uyumlu model aynı rastgele girdide
bit düzeyinde aynı logits üretti (`max_abs_error = 0,0`). Kaynak model hash'i
`b453d7c57423701e54e4dcdf4a08484716f671ac542f316cb0c156c8dfed2260`, Android
uyumlu model hash'i
`f4c93f2d7ded0e69af27e3ac0cd77ed7b8c9501c89e692910b77f8efac61de64`dır.
Yeniden paketleme `scripts/repack_tflite_inline_buffers.py` ile tekrarlanabilir.
