# Hespi v6 Eval10

This experiment keeps the frozen Hespi v5 EVAL set and changes only catalogue-number crop OCR.

## Change

- Select at most three high-confidence `number`, `barcode`, or `database_label` regions per sheet.
- Upscale each selected crop four times.
- Read both the enlarged RGB crop and an autocontrast grayscale crop.
- Use Tesseract PSM 7 and PSM 13 with a catalogue-number character whitelist.
- Rank and save every plausible candidate with its vote count and source readings.
- Present candidates to DeepSeek as hypotheses. Do not automatically promote the top candidate to truth.
- Send conflicting hypotheses to the review queue.

ZXing barcode decoding, GBIF authorship correction, DeepSeek settings, the shared image manifest, and all non-catalogue OCR channels remain unchanged.

## Run

Use **Actions -> Hespi v6 eval10 catalog number OCR ensemble -> Run workflow**, or update:

```text
.github/run-hespi-v6-eval10.trigger
```

Ordinary code pushes do not run this experiment.

## Decision rule

Do not expand beyond eval10 unless the run parses all ten outputs, improves catalogue-number exact match without reducing overall token F1, and passes the existing unsupported-prediction and review-rate gates.
