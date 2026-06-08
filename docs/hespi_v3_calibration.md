# Hespi v3 Five-Record Calibration

This workflow tests two decisions before another large paid evaluation:

1. whether DeepSeek non-thinking JSON mode preserves extraction quality while reducing tokens;
2. whether a lean hybrid prompt can omit small label-field crops without losing the useful catalogue-number and type-status evidence.

## Evaluation changes

The evaluator applies field-specific normalization:

- catalogue numbers ignore spaces and punctuation;
- common date formats are converted to ISO dates;
- collector-name token order and punctuation are normalized;
- coordinates use a small numeric tolerance;
- type-status values use a controlled status vocabulary.

Each flattened prediction also retains its `evidence_span`. The evaluator checks both whether that span occurs directly in OCR and whether the predicted value aligns with the span, then counts unsupported predictions. This is a direct-support diagnostic, not semantic entailment.

## Run modes

Open **Actions -> Hespi v3 five-record calibration -> Run workflow** and select one mode:

- `nonthinking_hybrid`: recommended first smoke test; five DeepSeek calls;
- `thinking_hybrid`: current thinking behavior on the same frozen sample;
- `nonthinking_lean`: whole sheet, primary label, and selected number/stamp components, without small label-field crops;
- `comparison`: runs all three variants, for fifteen DeepSeek calls.

Ordinary code pushes do not start the calibration workflow.

Maintainers can also start only the recommended `nonthinking_hybrid` smoke by updating:

```text
.github/run-hespi-v3-nonthinking.trigger
```

Other pushes do not start the calibration workflow.

The complete three-way comparison can be started by updating:

```text
.github/run-hespi-v3-comparison.trigger
```

and using `[run-hespi-v3-comparison]` in that commit message.

## Decision rules

Prefer non-thinking mode when:

- all five responses parse;
- total tokens fall by at least 75%;
- token F1 falls by no more than 0.02.

Prefer lean hybrid when:

- it stays within 0.01 token F1 of non-thinking full hybrid;
- prompt tokens fall by at least 25%;
- unsupported predictions do not increase.

Do not scale to 50 records when parse success is below 95% or unsupported prediction rate exceeds 15%.

The combined report is written to:

```text
reports/hespi_v3_calibration/hespi_v3_calibration_report.md
```
