# Dataset Card — SmartWaste Vision için RealWaste

[English version](DATASET_CARD.en.md)

## Özet

Bu proje [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) veri
setinin yerel olarak denetlenmiş altı sınıflı bir görünümünü kullanır. RealWaste,
Avustralya NSW Wollongong'daki Whyte's Gully Waste and Resource Recovery
Facility tesisine gelen gerçek atıkların görüntülerinden oluşur. Kaynak yayın:
[RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using
Deep Learning](https://www.mdpi.com/2078-2489/14/12/633), UCI DOI:
https://doi.org/10.24432/C5SS4G.

Ham veri 4.752 görsel ve dokuz kaynak sınıfı içerir. Dosya denetiminde
4.752/4.752 görsel açılabilir bulundu; ham görseller taşınmadı veya
değiştirilmedi.

## MVP sınıf eşlemesi

| Kaynak sınıf | Hedef sınıf | Görsel | Karar |
|---|---|---:|---|
| Plastic | `plastic` | 921 | Dahil |
| Glass | `glass` | 420 | Dahil |
| Metal | `metal` | 790 | Dahil |
| Paper + Cardboard | `paper_cardboard` | 961 | Birleştirildi |
| Food Organics | `organic` | 411 | Dahil |
| Miscellaneous Trash | `general_waste` | 495 | Dahil |
| Textile Trash | — | 318 | MVP dışında |
| Vegetation | — | 436 | MVP dışında |

Elektronik atık sınıfı RealWaste'ta bulunmadığı ve lisansı/kalitesi net ek bir
kaynak eklenmediği için MVP altı sınıfta tutuldu. Dahil edilen toplam 3.998,
dışlanan toplam 754 görseldir.

## Split ve kalibrasyon

Seed `20260803` ile deterministik, sınıf ve özgün kaynak sınıf bazında
stratified split:

| Split | Görsel |
|---|---:|
| Train | 2.800 |
| Validation | 600 |
| Test | 598 |

Bilinen aynı çekim çiftleri `group_id` ile tek split'e zorlandı. Group leakage
`0`, tam SHA-256 kopya sayısı `0`dır. Quantization deneyi için yalnız train
split'inden sınıf başına 50 olmak üzere 300 görüntülük ayrı calibration
manifesti seçildi; validation/test kalibrasyona girmedi.

- Split manifest SHA-256:
  `07dfa10bab6efa2f40d6d09437e8b4774e5494dff07bfd70aab6639cc52b5ae4`
- Calibration manifest SHA-256:
  `ca400a38ea1b967cf26a6566b82ed1b97b0c701e310035332bc02be1bd0a88aa`
- Ham envanter SHA-256:
  `116dba88a7234f88b3b744368f59c5d1488194e4412f8de767d7843c055742db`

Split ayrıntısı [split raporunda](02-split-results.md), dosya düzeyi kayıtlar
`data/manifests/realwaste_file_manifest.csv` ve
`data/manifests/realwaste_split.csv` dosyalarındadır.

## Lisans ve dağıtım

3 Ağustos 2026'da UCI metadata sayfası **CC BY 4.0**, indirilen paket README'si
ve veri sahibinin GitHub README'si ise **CC BY-NC-SA 4.0** belirtmektedir. Bu
çelişki çözülene kadar proje muhafazakâr biçimde CC BY-NC-SA 4.0 koşullarını
varsayar.

- Ham görüntüler Git deposuna, APK'ya veya Docker imajına eklenmez.
- Proje/veri “ticari kullanıma hazır” diye tanıtılmaz.
- Kamuya açık ağırlık dağıtımı ayrıca lisans incelemesi gerektirir.
- Kaynak ve yayın atfı korunmalıdır.

Ayrıntı [kaynak ve lisans kaydındadır](../../data/manifests/realwaste_source_record.md).
Bu belge hukuki görüş değildir.

## Temsiliyet, mahremiyet ve bilinen sınırlamalar

- Veri tek bir Avustralya tesisinden gelir; dünya geneli atık akışını temsil
  ettiği varsayılamaz.
- Sınıflar dengesizdir; kâğıt/karton ve plastik organikten daha fazladır.
- Atık, arka plan ve çekim koşulu arasındaki tesadüfi korelasyonlar modele
  taşınabilir.
- Veri kartı kişisel veri içermediğini garanti etmez; kamuya açık yeniden
  dağıtımdan önce görsel düzeyinde mahremiyet incelemesi gerekir.
- Altı sınıf bertaraf yöntemini veya geri dönüştürülebilirliği kesin olarak
  belirlemez; kirlenme ve yerel kurallar görüntü etiketinde yoktur.

Bağımsız, farklı telefonlarla çekilmiş ve farklı yerel ortamlardan gelen bir
saha test seti oluşturulmadan production genellemesi iddia edilmemelidir.

## Yeniden üretim ve sürümleme

```powershell
python scripts/build_realwaste_splits.py
```

Kaynak paket, sınıf haritası, seed veya grouping notu değişirse manifest hash'i
değişmeli; eski test sonucu yeni veri sürümüne taşınmamalıdır.
