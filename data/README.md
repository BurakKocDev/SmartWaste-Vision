# SmartWaste Vision — Yerel Veri Klasörleri

Veri setleri ve üretilmiş görseller Git deposuna eklenmez. Kaynak, lisans,
dosya hash'i ve sınıf eşlemesi `manifests/` altında sürümlenebilir kayıtlar olarak
tutulur.

## RealWaste indirme konumu

UCI'dan indirilen ZIP dosyasını veya açılmış veri seti klasörünü şu dizine koy:

```text
data/raw/realwaste/
```

İki kabul edilen örnek:

```text
data/raw/realwaste/realwaste.zip
```

veya:

```text
data/raw/realwaste/RealWaste/
├── Cardboard/
├── Food Organics/
├── Glass/
├── Metal/
├── Miscellaneous Trash/
├── Paper/
├── Plastic/
├── Textile Trash/
└── Vegetation/
```

Dosya veya klasör adı biraz farklıysa yeniden adlandırma. Veri doğrulama komutu
gerçek yapıyı tespit edip güvenli biçimde standartlaştıracaktır.

## Klasörlerin anlamı

- `raw/`: İndirildiği haliyle değiştirilmeyen kaynak veri
- `interim/`: Temizlik ve kalite kontrolünün ara çıktıları
- `processed/`: Eğitim için hazırlanmış manifest/split görünümü
- `external/`: Telefonda çekilecek bağımsız saha test görüntüleri
- `manifests/`: Kaynak, lisans, hash, sınıf ve split kayıtları

## Veri geldikten sonraki kontrol

Veri üzerinde değişiklik yapmadan önce aşağıdakiler doğrulanacaktır:

1. ZIP veya klasör yapısı ve toplam dosya sayısı
2. SHA-256 kaynak hash'i
3. Dokuz orijinal sınıf ve sınıf başına görüntü sayısı
4. Bozuk ve desteklenmeyen dosyalar
5. Tam ve algısal kopyalar
6. Lisans ve kaynak kaydı

Kaynak: https://archive.ics.uci.edu/dataset/908/realwaste

## Güncel denetim çıktıları

- `manifests/realwaste_file_manifest.csv`: Dosya, sınıf, boyut, çözünürlük,
  SHA-256, dHash ve pHash kayıtları
- `manifests/realwaste_audit.json`: Bütünlük, sınıf sayıları ve kopya adayları
- `manifests/realwaste_source_record.md`: Kaynak ve lisans çelişkisi kaydı
- `manifests/realwaste_review_notes.md`: Görsel denetim ve split notları

Denetim `scripts/audit_realwaste.py` ile ham veriyi değiştirmeden tekrar
çalıştırılabilir.
