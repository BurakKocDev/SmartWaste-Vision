# SmartWaste Vision portföy yayın kontrol listesi

## Hazır olanlar

- [x] Dondurulmuş test split'i ve altı sınıflı baseline
- [x] PyTorch FP32, ONNX FP32 ve LiteRT FP16/INT8 benchmark tablosu
- [x] Confusion matrix, Model Card ve Dataset Card
- [x] Türkçe ve İngilizce README
- [x] Modern Flutter kamera/galeri arayüzü
- [x] Düşük güvenli tahminde kontrollü geri çekilme
- [x] Android emülatörde offline debug ve release-mod inference smoke testi
- [x] AOT release-mod test APK ve paket içi model hash doğrulaması
- [x] Docker'lı PyTorch/FastAPI referans servisi
- [x] Portföy için ana ekran ve sonuç ekranı görselleri
- [x] Yenilenen UI ile final demo videosu
- [x] Türetilmiş model ağırlıklarını varsayılan olarak Git dışında tutan
  lisans-güvenli depo yapısı

## Kamuya açmadan önce

- [ ] Gerçek Android telefonda kamera, galeri ve uçak modu testi
- [ ] Aynı telefonda en az 50 warm inference ile ortalama/p95 model gecikmesi
- [ ] Uçtan uca gecikme ve peak/steady RAM ölçümü
- [ ] Production keystore ile imzalanmış AAB gerekiyorsa ayrı release yapılandırması
- [ ] RealWaste lisans çelişkisi nedeniyle veri ve model ağırlıklarının kamuya
  açık dağıtım kararı
- [ ] Depo lisansı ve GitHub görünürlüğü (private/public) kararı
- [ ] Gizli anahtar ve büyük dosya taraması

## Mevcut yerel artefaktlar

- Debug APK: `artifacts/mobile/smartwaste-vision-debug.apk`
- Release-mod test APK: `artifacts/mobile/smartwaste-vision-release.apk`
- Eski UI demo taslağı:
  `artifacts/demo/smartwaste-vision-demo-tr-draft-old-ui.mp4`
- GitHub'a eklenecek final demo: `docs/assets/smartwaste-vision-demo-tr.mp4`

`artifacts/` klasörü Git tarafından ignore edilir. Bu nedenle APK ve video, depo
yayınından ayrı bir GitHub Release veya portföy medya adımıyla yüklenmelidir.
Mobil TFLite ağırlığı da lisans kararına kadar ignore edilir; yerel geri yükleme
adımı `apps/mobile/assets/models/README.md` içinde belgelenmiştir.

## Demo akışı

1. Ana ekranda çevrimdışı ve gizlilik mesajını göster.
2. Galeriden net bir cam veya karton fotoğrafı seç.
3. Sınıf, güven, kutu rengi, model süresi ve ilk üç olasılığı göster.
4. Düşük güvenli plastik test fotoğrafıyla sarı uyarı durumunu göster.
5. Uçak modunu göstererek tahminin cihazda çalıştığını vurgula.
