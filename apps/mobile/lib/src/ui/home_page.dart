import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../domain/waste_category.dart';
import '../inference/classification_result.dart';
import '../inference/waste_classifier.dart';

const _deepGreen = Color(0xFF123D32);
const _brandGreen = Color(0xFF0B7A5B);
const _mint = Color(0xFFDDF3E9);
const _canvas = Color(0xFFF2F6F3);
const _ink = Color(0xFF13251F);
const _mutedInk = Color(0xFF60706A);

class SmartWasteHomePage extends StatefulWidget {
  const SmartWasteHomePage({
    super.key,
    required this.classifier,
    this.imagePicker,
  });

  final WasteClassifier classifier;
  final ImagePicker? imagePicker;

  @override
  State<SmartWasteHomePage> createState() => _SmartWasteHomePageState();
}

class _SmartWasteHomePageState extends State<SmartWasteHomePage> {
  late final ImagePicker _imagePicker;
  bool _isInitializing = true;
  bool _isBusy = false;
  String? _error;
  XFile? _selectedImage;
  ClassificationResult? _result;

  @override
  void initState() {
    super.initState();
    _imagePicker = widget.imagePicker ?? ImagePicker();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      await widget.classifier.initialize();
      if (!mounted) return;
      setState(() => _isInitializing = false);
      if (Platform.isAndroid) {
        await _recoverLostImage();
      }
    } catch (error, stackTrace) {
      debugPrint('SmartWaste model initialization failed: $error');
      debugPrintStack(stackTrace: stackTrace);
      if (!mounted) return;
      setState(() {
        _isInitializing = false;
        _error = 'Model başlatılamadı. Uygulamayı yeniden açmayı dene.';
      });
    }
  }

  Future<void> _recoverLostImage() async {
    final response = await _imagePicker.retrieveLostData();
    if (response.isEmpty || response.file == null || !mounted) return;
    await _classify(response.file!);
  }

  Future<void> _pickAndClassify(ImageSource source) async {
    if (_isInitializing || _isBusy) return;
    try {
      final image = await _imagePicker.pickImage(
        source: source,
        imageQuality: 95,
        preferredCameraDevice: CameraDevice.rear,
      );
      if (image != null) await _classify(image);
    } catch (error, stackTrace) {
      debugPrint('SmartWaste image picker failed: $error');
      debugPrintStack(stackTrace: stackTrace);
      if (!mounted) return;
      setState(() => _error = 'Fotoğraf alınamadı. İzinleri kontrol et.');
    }
  }

  Future<void> _classify(XFile image) async {
    setState(() {
      _selectedImage = image;
      _result = null;
      _error = null;
      _isBusy = true;
    });
    try {
      final result = await widget.classifier.classify(
        await image.readAsBytes(),
      );
      if (!mounted) return;
      setState(() => _result = result);
    } on FormatException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message.toString());
    } catch (error, stackTrace) {
      debugPrint('SmartWaste classification failed: $error');
      debugPrintStack(stackTrace: stackTrace);
      if (!mounted) return;
      setState(
        () => _error = kDebugMode
            ? 'Tahmin yapılamadı: $error'
            : 'Tahmin yapılamadı. Başka bir fotoğraf dene.',
      );
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  @override
  void dispose() {
    widget.classifier.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _canvas,
      body: SafeArea(
        bottom: false,
        child: LayoutBuilder(
          builder: (context, constraints) {
            return SingleChildScrollView(
              padding: const EdgeInsets.only(bottom: 30),
              child: Column(
                children: [
                  const _HeroHeader(),
                  Transform.translate(
                    offset: const Offset(0, -24),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 18),
                      child: Center(
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 680),
                          child: Column(
                            children: [
                              _CaptureCard(
                                image: _selectedImage,
                                isBusy: _isBusy,
                                isInitializing: _isInitializing,
                                onCamera: () =>
                                    _pickAndClassify(ImageSource.camera),
                                onGallery: () =>
                                    _pickAndClassify(ImageSource.gallery),
                              ),
                              if (_error != null) ...[
                                const SizedBox(height: 14),
                                _ErrorNotice(message: _error!),
                              ],
                              const SizedBox(height: 16),
                              AnimatedSwitcher(
                                duration: const Duration(milliseconds: 320),
                                switchInCurve: Curves.easeOutCubic,
                                switchOutCurve: Curves.easeInCubic,
                                child: _result == null
                                    ? const _ReadyPanel(
                                        key: ValueKey('ready-panel'),
                                      )
                                    : _ResultCard(
                                        key: ValueKey(
                                          '${_result!.best.category.canonicalLabel}-${_result!.isLowConfidence}',
                                        ),
                                        result: _result!,
                                      ),
                              ),
                              const SizedBox(height: 18),
                              const _PrivacyNotice(),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

class _HeroHeader extends StatelessWidget {
  const _HeroHeader();

  @override
  Widget build(BuildContext context) {
    final wide = MediaQuery.sizeOf(context).width >= 560;
    return Container(
      width: double.infinity,
      padding: EdgeInsets.fromLTRB(wide ? 34 : 20, 18, wide ? 34 : 20, 52),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF102F28), Color(0xFF0C644C)],
        ),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(36)),
      ),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 680),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  _BrandMark(),
                  SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'SmartWaste Vision',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 17,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -0.2,
                          ),
                        ),
                        Text(
                          'Akıllı atık rehberi',
                          style: TextStyle(
                            color: Color(0xFFB9D9CD),
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                  _OfflineBadge(),
                ],
              ),
              const SizedBox(height: 28),
              Text(
                'Atığını tara.\nDoğru kutuyu bul.',
                style: Theme.of(
                  context,
                ).textTheme.headlineLarge?.copyWith(color: Colors.white),
              ),
              const SizedBox(height: 11),
              const Text(
                'Fotoğraf cihazından çıkmadan saniyeler içinde sınıflandır.',
                style: TextStyle(
                  color: Color(0xFFD2E6DE),
                  fontSize: 15,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 20),
              const Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  _FeaturePill(icon: Icons.category_outlined, label: '6 sınıf'),
                  _FeaturePill(icon: Icons.bolt_rounded, label: 'Cihazda AI'),
                  _FeaturePill(
                    icon: Icons.lock_outline_rounded,
                    label: 'Gizli ve güvenli',
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BrandMark extends StatelessWidget {
  const _BrandMark();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 46,
      height: 46,
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: Colors.white.withValues(alpha: 0.16)),
      ),
      child: const Icon(Icons.recycling_rounded, color: Colors.white, size: 27),
    );
  }
}

class _OfflineBadge extends StatelessWidget {
  const _OfflineBadge();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: const Color(0xFFBDF2D8),
        borderRadius: BorderRadius.circular(999),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.offline_bolt_rounded, size: 15, color: _deepGreen),
          SizedBox(width: 5),
          Text(
            'Çevrimdışı',
            style: TextStyle(
              color: _deepGreen,
              fontSize: 12,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _FeaturePill extends StatelessWidget {
  const _FeaturePill({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: const Color(0xFFD5EEE4), size: 15),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              color: Color(0xFFE8F5F0),
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _CaptureCard extends StatelessWidget {
  const _CaptureCard({
    required this.image,
    required this.isBusy,
    required this.isInitializing,
    required this.onCamera,
    required this.onGallery,
  });

  final XFile? image;
  final bool isBusy;
  final bool isInitializing;
  final VoidCallback onCamera;
  final VoidCallback onGallery;

  @override
  Widget build(BuildContext context) {
    final enabled = !isBusy && !isInitializing;
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(30),
        boxShadow: const [
          BoxShadow(
            color: Color(0x1A14372C),
            blurRadius: 28,
            offset: Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        children: [
          AspectRatio(
            aspectRatio: 4 / 3,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(23),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (image != null)
                    Image.file(
                      File(image!.path),
                      fit: BoxFit.cover,
                      errorBuilder: (_, _, _) => const _EmptyPreview(),
                    )
                  else
                    const _EmptyPreview(),
                  Positioned(
                    top: 12,
                    left: 12,
                    child: _PreviewBadge(hasImage: image != null),
                  ),
                  if (isBusy || isInitializing)
                    _LoadingOverlay(isInitializing: isInitializing),
                ],
              ),
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  key: const Key('cameraButton'),
                  onPressed: enabled ? onCamera : null,
                  icon: const Icon(Icons.photo_camera_rounded, size: 20),
                  label: const Text('Kamera'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton.icon(
                  key: const Key('galleryButton'),
                  onPressed: enabled ? onGallery : null,
                  icon: const Icon(Icons.photo_library_outlined, size: 20),
                  label: const Text('Galeri'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _PreviewBadge extends StatelessWidget {
  const _PreviewBadge({required this.hasImage});

  final bool hasImage;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
      decoration: BoxDecoration(
        color: _deepGreen.withValues(alpha: 0.88),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            hasImage ? Icons.image_outlined : Icons.center_focus_strong_rounded,
            size: 14,
            color: Colors.white,
          ),
          const SizedBox(width: 5),
          Text(
            hasImage ? 'SEÇİLEN FOTOĞRAF' : 'AI HAZIR',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 10,
              fontWeight: FontWeight.w900,
              letterSpacing: 0.6,
            ),
          ),
        ],
      ),
    );
  }
}

class _EmptyPreview extends StatelessWidget {
  const _EmptyPreview();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFFE9F4EF), Color(0xFFDCEAE4)],
        ),
      ),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.82),
                shape: BoxShape.circle,
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x12143B2F),
                    blurRadius: 18,
                    offset: Offset(0, 7),
                  ),
                ],
              ),
              child: const Icon(
                Icons.document_scanner_outlined,
                color: _brandGreen,
                size: 32,
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'Bir atık fotoğrafı ekle',
              style: TextStyle(
                color: _ink,
                fontSize: 17,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 5),
            const Text(
              'Tek nesne • sade arka plan • iyi ışık',
              textAlign: TextAlign.center,
              style: TextStyle(color: _mutedInk, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}

class _LoadingOverlay extends StatelessWidget {
  const _LoadingOverlay({required this.isInitializing});

  final bool isInitializing;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: _deepGreen.withValues(alpha: 0.76),
      child: Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.10),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.white.withValues(alpha: 0.16)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 2.6,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                isInitializing ? 'Model hazırlanıyor' : 'Atık analiz ediliyor',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ReadyPanel extends StatelessWidget {
  const _ReadyPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFFDCE7E2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.auto_awesome_rounded, color: _brandGreen, size: 20),
              SizedBox(width: 8),
              Text(
                '6 atık türünü tanır',
                style: TextStyle(
                  color: _ink,
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 13),
          Wrap(
            spacing: 7,
            runSpacing: 7,
            children: WasteCategory.values
                .map((category) => _CategoryChip(category: category))
                .toList(growable: false),
          ),
        ],
      ),
    );
  }
}

class _CategoryChip extends StatelessWidget {
  const _CategoryChip({required this.category});

  final WasteCategory category;

  @override
  Widget build(BuildContext context) {
    final color = Color(category.binColorValue);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(
            category.turkishName,
            style: const TextStyle(
              color: _ink,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({super.key, required this.result});

  final ClassificationResult result;

  @override
  Widget build(BuildContext context) {
    final best = result.best;
    final category = best.category;
    final categoryColor = Color(category.binColorValue);
    final lowConfidence = result.isLowConfidence;
    final accent = lowConfidence ? const Color(0xFFE5A50A) : categoryColor;
    final surface = lowConfidence
        ? const [Color(0xFFFFFAEA), Color(0xFFFFF1C5)]
        : [Colors.white, categoryColor.withValues(alpha: 0.10)];

    return Semantics(
      liveRegion: true,
      label:
          '${lowConfidence ? 'Emin değilim' : category.turkishName}, güven yüzde ${(best.confidence * 100).round()}, ${category.binColorName} kutu',
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: surface,
          ),
          borderRadius: BorderRadius.circular(28),
          border: Border.all(color: accent.withValues(alpha: 0.48)),
          boxShadow: [
            BoxShadow(
              color: accent.withValues(alpha: 0.10),
              blurRadius: 24,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _StatusEyebrow(lowConfidence: lowConfidence, accent: accent),
            const SizedBox(height: 16),
            Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                _ConfidenceRing(value: best.confidence, color: accent),
                const SizedBox(width: 17),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        lowConfidence
                            ? 'Bir kez daha deneyelim'
                            : category.turkishName,
                        style: Theme.of(
                          context,
                        ).textTheme.headlineSmall?.copyWith(color: _ink),
                      ),
                      const SizedBox(height: 5),
                      Text(
                        lowConfidence
                            ? 'En yakın eşleşme: ${category.turkishName}'
                            : category.guidance,
                        style: const TextStyle(
                          color: _mutedInk,
                          fontSize: 14,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 19),
            Row(
              children: [
                Expanded(
                  child: _MetricTile(
                    icon: Icons.delete_outline_rounded,
                    label: 'Kutu rengi',
                    value: category.binColorName,
                    accent: categoryColor,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _MetricTile(
                    icon: Icons.speed_rounded,
                    label: 'Model süresi',
                    value: '${result.inferenceDuration.inMilliseconds} ms',
                    accent: _brandGreen,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            const Text(
              'Tahmin dağılımı',
              style: TextStyle(
                color: _ink,
                fontSize: 13,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 10),
            ...result.ranking
                .take(3)
                .map(
                  (score) => Padding(
                    padding: const EdgeInsets.only(bottom: 9),
                    child: _PredictionBar(score: score),
                  ),
                ),
            if (lowConfidence) ...[
              const SizedBox(height: 5),
              Container(
                padding: const EdgeInsets.all(13),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.68),
                  borderRadius: BorderRadius.circular(15),
                ),
                child: const Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.lightbulb_outline_rounded,
                      size: 19,
                      color: Color(0xFF7A5A00),
                    ),
                    SizedBox(width: 9),
                    Expanded(
                      child: Text(
                        'Atığı yaklaştırıp sade bir arka planda yeniden dene.',
                        style: TextStyle(
                          color: Color(0xFF6B520E),
                          fontSize: 12,
                          height: 1.35,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _StatusEyebrow extends StatelessWidget {
  const _StatusEyebrow({required this.lowConfidence, required this.accent});

  final bool lowConfidence;
  final Color accent;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: accent.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            lowConfidence
                ? Icons.help_outline_rounded
                : Icons.check_circle_outline_rounded,
            size: 15,
            color: lowConfidence ? const Color(0xFF785500) : _deepGreen,
          ),
          const SizedBox(width: 6),
          Text(
            lowConfidence ? 'DÜŞÜK GÜVEN' : 'EŞLEŞME BULUNDU',
            style: TextStyle(
              color: lowConfidence ? const Color(0xFF785500) : _deepGreen,
              fontSize: 10,
              fontWeight: FontWeight.w900,
              letterSpacing: 0.7,
            ),
          ),
        ],
      ),
    );
  }
}

class _ConfidenceRing extends StatelessWidget {
  const _ConfidenceRing({required this.value, required this.color});

  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 84,
      height: 84,
      child: Stack(
        alignment: Alignment.center,
        children: [
          SizedBox.expand(
            child: CircularProgressIndicator(
              value: value,
              strokeWidth: 8,
              strokeCap: StrokeCap.round,
              color: color,
              backgroundColor: color.withValues(alpha: 0.16),
            ),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '%${(value * 100).toStringAsFixed(0)}',
                style: const TextStyle(
                  color: _ink,
                  fontSize: 20,
                  fontWeight: FontWeight.w900,
                  letterSpacing: -0.5,
                ),
              ),
              const Text(
                'GÜVEN',
                style: TextStyle(
                  color: _mutedInk,
                  fontSize: 8,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 0.7,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  const _MetricTile({
    required this.icon,
    required this.label,
    required this.value,
    required this.accent,
  });

  final IconData icon;
  final String label;
  final String value;
  final Color accent;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.78),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE5ECE8)),
      ),
      child: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: accent.withValues(alpha: 0.13),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: accent, size: 19),
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: _mutedInk, fontSize: 10),
                ),
                const SizedBox(height: 1),
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: _ink,
                    fontSize: 13,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PredictionBar extends StatelessWidget {
  const _PredictionBar({required this.score});

  final ClassScore score;

  @override
  Widget build(BuildContext context) {
    final color = Color(score.category.binColorValue);
    return Row(
      children: [
        SizedBox(
          width: 94,
          child: Text(
            score.category.turkishName,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              color: _mutedInk,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        const SizedBox(width: 7),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(99),
            child: LinearProgressIndicator(
              value: score.confidence,
              minHeight: 7,
              color: color,
              backgroundColor: const Color(0xFFE4EBE7),
            ),
          ),
        ),
        const SizedBox(width: 8),
        SizedBox(
          width: 38,
          child: Text(
            '%${(score.confidence * 100).toStringAsFixed(1)}',
            textAlign: TextAlign.end,
            style: const TextStyle(
              color: _ink,
              fontSize: 10,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
      ],
    );
  }
}

class _ErrorNotice extends StatelessWidget {
  const _ErrorNotice({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFE6E2),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFFFB5AA)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline_rounded, color: Color(0xFF9D2F21)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(color: Color(0xFF81261B), fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}

class _PrivacyNotice extends StatelessWidget {
  const _PrivacyNotice();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: _mint.withValues(alpha: 0.58),
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.lock_outline_rounded, size: 16, color: _deepGreen),
          SizedBox(width: 7),
          Flexible(
            child: Text(
              'Tahmin cihazında yapılır; fotoğraf hiçbir sunucuya gönderilmez.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: _deepGreen,
                fontSize: 11,
                height: 1.35,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
