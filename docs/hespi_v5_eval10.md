# Hespi v5 Eval10

This experiment targets two failure modes observed in GitHub Actions run `27160844627`.

## Changes

- ZXing-C++ decodes machine-readable barcodes once per sheet and appends them as catalog-number evidence.
- Multiple decoded barcodes are explicitly marked ambiguous.
- DeepSeek is told not to confuse job, collection, or image-processing numbers with the specimen catalog number.
- GBIF species matching may correct scientific-name authorship only when:
  - match confidence is at least 90;
  - the parsed binomial is unchanged.
- The original LLM value and every GBIF response field used for correction remain in `reconciliation.csv`.

## Run

Use **Actions -> Hespi v5 eval10 barcode and GBIF -> Run workflow**, or update:

```text
.github/run-hespi-v5-eval10.trigger
```

Ordinary code pushes do not run this experiment.
