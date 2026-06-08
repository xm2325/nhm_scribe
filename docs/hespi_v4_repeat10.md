# Hespi v4 Eval10 Three-Repeat Stability

This experiment runs layout detection and OCR once, then sends the exact same ten prompts to DeepSeek three times.

## Purpose

- measure non-thinking model variability;
- distinguish OCR/input variation from model-output variation;
- build a consensus prediction and human-review queue;
- prevent unsupported high-risk fields from being accepted automatically.

## Evidence and review statuses

Predicted fields receive one evidence status:

- `direct`: the evidence span occurs in OCR and aligns with the prediction;
- `partial_direct`: partial OCR support with prediction alignment;
- `prediction_in_ocr`: the predicted value occurs in OCR but the supplied evidence span is inadequate;
- `contextual_inference`: the evidence span occurs in OCR but does not directly support the prediction;
- `unsupported`: no adequate OCR support;
- `not_predicted`: the field is empty.

Only unanimous predictions with direct evidence in all three repeats and confidence of at least 0.75 are eligible for automatic acceptance. High-risk fields use a 0.90 confidence review threshold.

## Run

Use **Actions -> Hespi v4 eval10 repeated lean stability -> Run workflow**, or update:

```text
.github/run-hespi-v4-repeat10.trigger
```

Ordinary pushes do not start this workflow.

The combined report is:

```text
reports/hespi_v4_repeat10/hespi_v4_repeat10_report.md
```
