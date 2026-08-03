# RealWaste — İlk Veri Denetimi Notları

- Denetim tarihi: 3 Ağustos 2026
- Araç: `scripts/audit_realwaste.py`
- Sonuç: Dosya bütünlüğü geçti; lisans incelemesi açık

## Bütünlük özeti

| Ölçüt | Sonuç |
|---|---:|
| Görüntü | 4.752 |
| Sınıf | 9 |
| Çözünürlük | Tamamı 524×524 |
| Bozuk görüntü | 0 |
| Desteklenmeyen dosya | 0 |
| Birebir SHA-256 kopya grubu | 0 |
| dHash mesafesi ≤2 aday çifti | 2 |
| pHash mesafesi ≤6 aday çifti | 72 |

Sınıf sayıları kaynak README ve UCI metadata ile birebir uyuşmaktadır.

## Görsel olarak incelenen adaylar

### Aynı çekim/nesne grubu olarak tutulacak

- `Food Organics/Food Organics_81.jpg`
- `Food Organics/Food Organics_82.jpg`

Bu iki görüntü aynı sahnenin çok küçük farklılık taşıyan ardışık çekimleridir.
Train/validation/test arasında bölünmemelidir.

- `Metal/Metal_655.jpg`
- `Metal/Metal_658.jpg`

İki görüntü aynı türde, büyük olasılıkla aynı metal nesnenin yakın çekimleridir.
Muhafazakâr split yaklaşımıyla aynı grupta tutulmalıdır.

### Yanlış algısal hash alarmı

- `Textile Trash/Textile Trash_172.jpg` ile `Metal/Metal_126.jpg`
- `Plastic/Plastic_218.jpg` ile `Plastic/Plastic_216.jpg`
- `Plastic/Plastic_763.jpg` ile `Plastic/Plastic_204.jpg`

Bu görüntüler farklı nesnelerdir. Benzer açık fon ve merkezdeki koyu/yuvarlak
nesne, düşük çözünürlüklü algısal hash'i yanıltmıştır.

## Split kararı

- Algısal hash adayları otomatik olarak silinmez.
- Aynı çekim olduğu doğrulanan örnekler bir `group_id` ile aynı split'e atanır.
- 72 pHash adayı, split manifesti üretilmeden önce embedding benzerliği ve
  örneklem bazlı görsel denetimle ikinci kez incelenir.
- Harici telefon test seti RealWaste split'lerinden tamamen bağımsız tutulur.

## Üretilen dosyalar

- `data/manifests/realwaste_file_manifest.csv`
- `data/manifests/realwaste_audit.json`
- `data/manifests/realwaste_source_record.md`
