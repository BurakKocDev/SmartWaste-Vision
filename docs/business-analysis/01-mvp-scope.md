# SmartWaste Vision — MVP Kapsamı

## Problem

Kullanıcılar tek bir atığın hangi genel malzeme kategorisine ait olduğunu ve
hangi renk biriktirme ekipmanıyla ilişkilendirildiğini her zaman bilmiyor.
SmartWaste Vision, mobil cihazda ağ bağlantısı olmadan genel bir sınıf önerisi
sunarak kullanıcıyı yönlendirir.

Uygulama bir mevzuat veya tehlikeli atık karar sistemi değildir. Sonuç, görsel
benzerliğe dayalı genel yönlendirmedir.

## Birincil kullanım senaryosu

1. Kullanıcı sade bir arka plan önünde tek bir atığı kameraya gösterir veya
   galeriden seçer.
2. Uygulama fotoğrafı cihaz üzerinde işler.
3. En olası atık sınıfını ve güven skorunu gösterir.
4. Türkiye Sıfır Atık renk skalasına göre renk ve metin etiketi gösterir.
5. Güven eşiğin altındaysa “Emin değilim; farklı açı veya ışıkla tekrar dene”
   mesajını gösterir.

## Girdi sözleşmesi

- Görüntüde tek bir baskın atık nesnesi bulunmalıdır.
- Nesne kadrajın anlamlı bölümünü kaplamalıdır.
- Çoklu atık yığını, çok küçük nesne, yoğun örtüşme veya aşırı bulanıklık
  desteklenmez.
- Model bir sınıflandırıcıdır; nesnenin yerini bulmaz ve birden fazla nesneyi
  ayrı ayrı saymaz.

## Hedef sınıflar ve renkler

| Model etiketi | Türkçe ad | Biriktirme rengi |
|---|---|---|
| `plastic` | Plastik | Sarı |
| `glass` | Cam | Yeşil |
| `metal` | Metal | Açık gri |
| `paper_cardboard` | Kâğıt/karton | Mavi |
| `organic` | Organik | Kahverengi |
| `electronic` | Elektronik | Mor |
| `general` | Evsel nitelikli diğer/genel atık | Koyu gri |

Renk tek başına bilgi taşımaz; sınıf adı ve erişilebilir piktogram da gösterilir.
Uygulama uygulama zamanındaki resmi renk kodlarını kullanır. Yerel toplama
kuralları farklı olabileceği için ekranda “Yerel kuralları kontrol edin” notu
bulunur.

Elektronik sınıfı veri kapısını geçmezse ilk sürüm altı sınıfla çıkar. Bu durum
5–8 sınıflı kabul edilmiş MVP sınırı içindedir.

## MVP içinde

- MobileNetV3-Small transfer learning
- Tek bir eğitilmiş checkpoint ve beş dağıtım çıktısı
- Accuracy, macro F1, sınıf bazlı metrikler ve confusion matrix
- Boyut, ortalama/p95 gecikme, yükleme süresi ve RAM ölçümü
- Windows/x86-64 CPU referans benchmark'ı
- Gerçek Android telefonda LiteRT benchmark'ı
- Flutter kamera ve galeri akışı
- Çevrimdışı tahmin
- Güven skoru ve “emin değilim” davranışı
- Docker'lı küçük PyTorch/FastAPI referans API'si
- Türkçe ve İngilizce portföy belgeleri

## MVP dışında

- Object detection, segmentation ve çoklu nesne
- Canlı kamera inference'ı veya video
- Robotik ayırma ve geri dönüşüm bandı
- Harita, konum veya geri dönüşüm noktaları
- Hesap, oturum, puan ve rozet
- Yönetim paneli ve tahmin geçmişi
- Bulut zorunluluğu ve kullanıcı görseli saklama
- Tehlikeli atık için kesin güvenlik kararı

## Başlangıç kabul hedefleri

Bu değerler baseline sonuçları değildir; ilk ölçüm sonrasında kanıta dayalı
olarak revize edilebilir.

| Ölçüt | Hedef |
|---|---:|
| Harici test accuracy | En az 0,82 |
| Harici test macro F1 | En az 0,80 |
| En düşük sınıf recall | En az 0,70 |
| FP export top-1 uyumu | PyTorch'a göre en az %99 |
| INT8 export top-1 uyumu | PyTorch'a göre en az %97 |
| Quantization macro F1 kaybı | En fazla 2 yüzde puanı |
| PyTorch/ONNX FP32 boyutu | En fazla 15 MB |
| TFLite FP16 boyutu | En fazla 7 MB |
| ONNX/TFLite INT8 boyutu | En fazla 4 MB |
| Android inference p95 | En fazla 250 ms, batch 1, 224×224 |
| Android soğuk olmayan uçtan uca p95 | En fazla 500 ms |
| Çevrimdışı çalışma | Uçak modunda %100 |

RAM için ilk gün hedef cihaz baseline'ı ölçülür. Kabul hedefi, uygulamanın
inference sırasında ölçülen tepe PSS artışının 150 MB'ı aşmamasıdır.

## Kullanıcı kabul senaryoları

- Desteklenen bir atığın iyi ışıklı fotoğrafı sınıf, güven ve renk gösterir.
- Düşük güvenli fotoğraf kesin kutu önerisi üretmez.
- Birden fazla atık içeren fotoğrafta kullanıcı tek nesne çekmeye yönlendirilir.
- Kamera izni reddedilirse galeri akışı kullanılabilir.
- Ağ kapalıyken tahmin sonucu aynıdır.
- Sonuç yalnız renkle değil metin ve simgeyle de anlaşılır.
