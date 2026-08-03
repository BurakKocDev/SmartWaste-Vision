enum WasteCategory {
  plastic(
    canonicalLabel: 'plastic',
    turkishName: 'Plastik',
    binColorName: 'Sarı',
    binColorValue: 0xFFF7C600,
    guidance: 'Sarı plastik atık kutusuna bırak.',
  ),
  glass(
    canonicalLabel: 'glass',
    turkishName: 'Cam',
    binColorName: 'Yeşil',
    binColorValue: 0xFF238636,
    guidance: 'Yeşil cam atık kutusuna bırak.',
  ),
  metal(
    canonicalLabel: 'metal',
    turkishName: 'Metal',
    binColorName: 'Açık gri',
    binColorValue: 0xFFB0BEC5,
    guidance: 'Açık gri metal atık kutusuna bırak.',
  ),
  paperCardboard(
    canonicalLabel: 'paper_cardboard',
    turkishName: 'Kâğıt / karton',
    binColorName: 'Mavi',
    binColorValue: 0xFF1976D2,
    guidance: 'Mavi kâğıt ve karton kutusuna bırak.',
  ),
  organic(
    canonicalLabel: 'organic',
    turkishName: 'Organik',
    binColorName: 'Kahverengi',
    binColorValue: 0xFF795548,
    guidance: 'Kahverengi organik atık kutusuna bırak.',
  ),
  generalWaste(
    canonicalLabel: 'general_waste',
    turkishName: 'Genel atık',
    binColorName: 'Koyu gri',
    binColorValue: 0xFF455A64,
    guidance: 'Koyu gri diğer atık kutusuna bırak.',
  );

  const WasteCategory({
    required this.canonicalLabel,
    required this.turkishName,
    required this.binColorName,
    required this.binColorValue,
    required this.guidance,
  });

  final String canonicalLabel;
  final String turkishName;
  final String binColorName;
  final int binColorValue;
  final String guidance;

  static WasteCategory fromCanonicalLabel(String label) {
    return values.firstWhere(
      (category) => category.canonicalLabel == label,
      orElse: () => throw ArgumentError.value(label, 'label', 'Unknown label'),
    );
  }
}
