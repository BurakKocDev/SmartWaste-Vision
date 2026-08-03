# Local mobile model asset

`smartwaste_fp16.tflite` is intentionally ignored by Git while the conflicting
RealWaste licence records are being resolved. The local Flutter build expects
the file at this directory with the following SHA-256:

```text
f4c93f2d7ded0e69af27e3ac0cd77ed7b8c9501c89e692910b77f8efac61de64
```

After running the documented training/export/repacking pipeline, restore the
local app asset from the accepted Android-compatible artifact:

```powershell
Copy-Item `
  artifacts\export\baseline\smartwaste_fp16_android.tflite `
  apps\mobile\assets\models\smartwaste_fp16.tflite
```

Verify it before building:

```powershell
(Get-FileHash `
  apps\mobile\assets\models\smartwaste_fp16.tflite `
  -Algorithm SHA256).Hash.ToLowerInvariant()
```

The tracked `model_contract.json` records the same hash, label order,
preprocessing contract, and confidence threshold.
