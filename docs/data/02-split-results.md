# RealWaste — Altı Sınıflı Split Sonucu

## Sonuç

Altı sınıflı 3.998 görüntü, seed `20260803` ile deterministik ve sınıf/orijinal
sınıf bazında stratified olarak ayrıldı.

| Split | Görüntü | Oran |
|---|---:|---:|
| Train | 2.800 | %70,04 |
| Validation | 600 | %15,01 |
| Test | 598 | %14,96 |

`Textile Trash` içindeki 318 ve `Vegetation` içindeki 436 görüntü MVP dışında
bırakıldı. Ham dosyalar taşınmadı veya değiştirilmedi; split yalnız manifestte
tutuluyor.

## Hedef sınıf dağılımı

| Sınıf | Train | Validation | Test | Toplam |
|---|---:|---:|---:|---:|
| `plastic` | 645 | 138 | 138 | 921 |
| `glass` | 294 | 63 | 63 | 420 |
| `metal` | 553 | 119 | 118 | 790 |
| `paper_cardboard` | 673 | 144 | 144 | 961 |
| `organic` | 288 | 62 | 61 | 411 |
| `general_waste` | 347 | 74 | 74 | 495 |

`paper_cardboard` ayrılırken `Paper` ve `Cardboard` kendi içlerinde de
stratified tutuldu.

## Sızıntı koruması

Görsel incelemede aynı çekim olduğu doğrulanan aşağıdaki çiftler tek `group_id`
ile aynı split'e zorlandı:

- `Food Organics_81.jpg` ve `Food Organics_82.jpg`
- `Metal_655.jpg` ve `Metal_658.jpg`

Group leakage sayısı `0`, tam SHA-256 kopya sayısı `0`dır. Diğer pHash adayları
otomatik olarak aynı nesne kabul edilmedi.

## INT8 calibration manifesti

Yalnız train split'inden, her hedef sınıftan 50 görüntü seçildi:

- Toplam: 300
- Validation/test kullanımı: 0
- Seçim: seed ve dosya yolu hash'ine göre deterministik

Bu manifest canary sonrası gerçek baseline/QAT optimizasyonunda kullanılacaktır.

## Yeniden üretim

```powershell
python scripts/build_realwaste_splits.py
```

Üretilen dosyalar:

- `data/manifests/realwaste_split.csv`
- `data/manifests/realwaste_split_summary.json`
- `data/manifests/int8_calibration.csv`

Split manifest SHA-256:
`07dfa10bab6efa2f40d6d09437e8b4774e5494dff07bfd70aab6639cc52b5ae4`
