# SmartWaste Vision — Veri ve Sınıf Planı

## Veri kararı

Birincil aday UCI Machine Learning Repository üzerindeki RealWaste veri setidir.
RealWaste 4.752 gerçek ortam görüntüsü ve dokuz sınıf sunar. Ancak lisans kaydı
çelişkilidir: UCI metadata sayfası CC BY 4.0, indirilen paketin README'si ve
veri sahibinin GitHub deposu CC BY-NC-SA 4.0 belirtmektedir.

Kaynak: https://archive.ics.uci.edu/dataset/908/realwaste

Lisans çelişkisi yazılı olarak netleşene kadar daha kısıtlayıcı
CC BY-NC-SA 4.0 koşulları esas alınır. Proje ticari kullanıma uygun diye
sunulmaz; veri veya model ağırlıklarının kamuya açık lisansı ayrıca incelenir.
Kaynak kaydı `data/manifests/realwaste_source_record.md` dosyasındadır.

TrashNet kolay bir baseline olsa da görüntülerin büyük bölümü beyaz fon üzerinde
çekilmiştir ve organik/elektronik sınıfları yoktur. TrashBox elektronik sınıfı
bakımından çekicidir; ancak görüntüler web'den kazınmıştır ve veri lisansı açık
değildir. Lisansı belirsiz görüntüler portföy modeline eklenmeyecektir.

## RealWaste sınıf eşlemesi

| RealWaste sınıfı | MVP sınıfı | Karar |
|---|---|---|
| Plastic | `plastic` | Doğrudan kullan |
| Glass | `glass` | Doğrudan kullan |
| Metal | `metal` | Doğrudan kullan |
| Paper | `paper_cardboard` | Cardboard ile birleştir |
| Cardboard | `paper_cardboard` | Paper ile birleştir |
| Food Organics | `organic` | Doğrudan kullan |
| Miscellaneous Trash | `general_waste` | Doğrudan kullan |
| Textile Trash | — | MVP dışında bırak |
| Vegetation | — | Organik tanımını gereksiz genişletmemek için dışarıda bırak |

Bu eşleme 3.998 görüntülük altı sınıflı taban sağlar. Kullanım koşulları yukarıda
belirtilen lisans çelişkisine tabidir.

## Elektronik sınıfı veri kapısı

Elektronik sınıfı aşağıdaki şartların tümünü sağlamalıdır:

- Görüntülerin yeniden dağıtım ve model eğitimi lisansı açıkça yazılıdır.
- Kaynak dosya ve lisans metni veri kataloğuna kaydedilir.
- En az 300 farklı özgün görüntü bulunur; basit augmentation görüntü sayılmaz.
- Telefon, kablo, küçük ev aleti ve devre/kart gibi alt tür çeşitliliği vardır.
- Görüntüler tek bir arka plan veya kaynak filigranıyla kolayca ayırt edilmez.
- En az 50 bağımsız telefon fotoğrafı yalnız harici test için ayrılır.

Kapı geçilmezse `electronic` etiketi ilk sürümden çıkarılır. Elektroniği korumak
için lisansı belirsiz arama motoru/Kaggle kopyaları birleştirilmez.

## Veri yaşam döngüsü

1. Ham veri değiştirilemez `data/raw/<source>/<class>` altında tutulur ve Git'e
   eklenmez.
2. Her dosya için SHA-256, kaynak, lisans, orijinal etiket, hedef etiket,
   çözünürlük ve durum manifestte tutulur.
3. Bozuk dosyalar, tam kopyalar ve algısal olarak çok yakın kopyalar işaretlenir.
4. Split, kaynak ve benzer görüntü grupları dikkate alınarak deterministik yapılır.
5. Train/validation/test oranı başlangıçta %70/%15/%15'tir.
6. Confidence eşiği yalnız validation setiyle seçilir.
7. Harici test seti farklı telefon, arka plan ve çekim oturumlarından oluşur;
   eğitim veya calibration için kullanılmaz.
8. INT8 calibration örnekleri yalnız train split'inden, sınıf dengeli ve
   tekrarlanabilir şekilde seçilir.

## Etiketleme kuralları

- Bir görüntüde birden fazla belirgin malzeme varsa baskınlık açık değilse
  `reject_ambiguous` olarak işaretlenir.
- Kirli veya ezilmiş nesneler gerçek kullanım çeşitliliği olarak korunabilir.
- Kompozit ambalajlar, pil, ampul, tıbbi atık ve tehlikeli atık ilk MVP sınıf
  sözleşmesine zorla sokulmaz.
- `general_waste`, diğer sınıfların belirsiz örnekleri için kaçış etiketi değildir;
  gerçekten evsel nitelikli diğer atık örneklerini içerir.
- Sınıf kararı yalnız dosya adına güvenilmeden örneklem bazlı görsel denetimle
  doğrulanır.

## Dataset Card içinde raporlanacaklar

- Kaynak, DOI/sürüm, indirme tarihi ve lisans
- Orijinal ve dönüştürülmüş sınıf dağılımı
- Dahil/dışarıda bırakılan sınıflar ve gerekçeleri
- Temizlik, deduplication ve split yöntemi
- Bilinen sahne, coğrafya, cihaz ve arka plan önyargıları
- Paylaşılabilen ve paylaşılamayan dosyalar
- Harici test setinin nasıl üretildiği
