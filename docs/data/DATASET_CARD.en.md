# Dataset Card — RealWaste for SmartWaste Vision

[Türkçe sürüm](DATASET_CARD.md)

## Summary

This project uses a locally audited six-class view of
[RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste). RealWaste was
collected from waste received at the Whyte's Gully Waste and Resource Recovery
Facility in Wollongong, NSW, Australia. Source publication:
[RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using
Deep Learning](https://www.mdpi.com/2078-2489/14/12/633); UCI DOI:
https://doi.org/10.24432/C5SS4G.

The source contains 4,752 images in nine classes. The local audit successfully
decoded all 4,752 files; raw images were not moved or modified.

## MVP taxonomy

| Source class | Target class | Images | Decision |
|---|---|---:|---|
| Plastic | `plastic` | 921 | Included |
| Glass | `glass` | 420 | Included |
| Metal | `metal` | 790 | Included |
| Paper + Cardboard | `paper_cardboard` | 961 | Merged |
| Food Organics | `organic` | 411 | Included |
| Miscellaneous Trash | `general_waste` | 495 | Included |
| Textile Trash | — | 318 | Excluded from MVP |
| Vegetation | — | 436 | Excluded from MVP |

RealWaste has no electronic-waste class, and no second source with sufficiently
clear licensing and quality was introduced. The MVP therefore contains 3,998
included and 754 excluded images across six classes.

## Splits and calibration

Deterministic seed `20260803`, stratified by target and original source class:

| Split | Images |
|---|---:|
| Train | 2,800 |
| Validation | 600 |
| Test | 598 |

Known same-capture pairs were forced into one split using `group_id`. Group
leakage is `0`; exact SHA-256 duplicate count is `0`. A 300-image quantization
calibration manifest selects 50 images per target class from train only; no
validation or test image is included.

- Split manifest SHA-256:
  `07dfa10bab6efa2f40d6d09437e8b4774e5494dff07bfd70aab6639cc52b5ae4`
- Calibration manifest SHA-256:
  `ca400a38ea1b967cf26a6566b82ed1b97b0c701e310035332bc02be1bd0a88aa`
- Raw inventory SHA-256:
  `116dba88a7234f88b3b744368f59c5d1488194e4412f8de767d7843c055742db`

See the [split report](02-split-results.md) and the file-level manifests under
`data/manifests`.

## License and distribution

On 3 August 2026, UCI metadata stated **CC BY 4.0**, while both the downloaded
README and the owner's GitHub README stated **CC BY-NC-SA 4.0**. Until this is
resolved, the project conservatively assumes the CC BY-NC-SA 4.0 terms.

- Raw images are excluded from Git, the APK, and Docker images.
- The project/data must not be advertised as commercially cleared.
- Public model-weight distribution requires a separate license review.
- Source and publication attribution must be preserved.

See the full [source and license record](../../data/manifests/realwaste_source_record.md).
This is not legal advice.

## Representation, privacy, and known limitations

- One Australian facility cannot be assumed to represent global waste streams.
- The class distribution is imbalanced; paper/cardboard and plastic outnumber
  organic examples.
- Spurious correlations between waste, backgrounds, and capture conditions may
  transfer to the model.
- This card does not guarantee that no personal data appears in any image; an
  image-level privacy review is required before public redistribution.
- A material class does not determine recyclability or disposal: contamination
  and local collection rules are not encoded in the labels.

Production generalisation must not be claimed before evaluation on an
independent field set captured with varied phones and local environments.

## Reproduction and versioning

```powershell
python scripts/build_realwaste_splits.py
```

A changed source package, taxonomy, seed, or grouping note must produce a new
manifest hash. Old test results must not be carried over to the new data version.
