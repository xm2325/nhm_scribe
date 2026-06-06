# Hespi v2 Strict Paired Evaluation

This experiment corrects the two main problems found in the first Hespi-lite POC:

1. baseline and candidate pipelines did not always use the same downloaded images;
2. DeepSeek was called even when no OCR evidence was available.

## Design

The acquisition stage samples 35 candidate EVAL records, downloads with retry handling for HTTP 429 and 5xx responses, validates each image, and computes SHA-256. It then uses fixed-seed stratified sampling to freeze 20 successfully downloaded EVAL records.

The three extraction pipelines only read the frozen manifest:

- **A full-image:** whole sheet, Tesseract `--psm 11`;
- **B field-only:** Hespi label fields, Tesseract `--psm 7`;
- **C hybrid:** whole sheet + primary label + field crops + selected catalog/type components.

Primary labels use `--psm 6`. Tight field crops receive 8% padding and a white border.

## Evidence gate

DeepSeek is skipped when:

- the frozen image is unavailable or not paired eligible;
- OCR text is empty;
- OCR contains fewer than 10 alphanumeric characters.

Skipped rows remain visible in diagnostics with `llm_call_attempted=false` and a specific `skip_reason`.

## Run

Use **Actions -> Hespi v2 strict paired evaluation -> Run workflow**.

The combined report is:

```text
reports/hespi_v2/hespi_v2_paired_report.md
```

It reports data availability, SHA-256 paired count, field macro and micro metrics, correctness among filled fields, OCR characters by evidence channel, skipped calls, and DeepSeek token cost.
