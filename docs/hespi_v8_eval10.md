# Hespi v8 Eval10

This experiment tests a supplementary handwritten-text-recognition path on the frozen ten-record EVAL set.

## HTR route

- Enable Hespi label-field detection only for `collector`, `year`, `month`, and `day`.
- Keep Tesseract output for every selected crop.
- Add a second reading from `microsoft/trocr-small-handwritten`.
- Pin Hugging Face Transformers below version 5 because Hespi 0.6.2 uses the version-4 TrOCR API.
- Send both readings to DeepSeek with explicit source headers.
- Treat TrOCR text as a hypothesis, not an automatic replacement.

Whole-sheet OCR, primary-label OCR, barcode decoding, catalogue-number OCR ensemble, the conservative catalogue resolver, GBIF checks, and DeepSeek settings remain unchanged from Hespi v7.

The report compares:

- HTR load and non-empty output rates;
- paired Tesseract and TrOCR text for each crop;
- collector and date-component evidence proxies;
- final `recordedBy` and `eventDate` metrics;
- overall extraction metrics and review rate.

## Run

Use **Actions -> Hespi v8 eval10 supplementary handwriting recognition -> Run workflow**, or update:

```text
.github/run-hespi-v8-eval10.trigger
```

Ordinary code pushes do not run this experiment.
