# Hespi-lite POC

This POC tests one narrow hypothesis: herbarium-specific label and field detection may improve OCR evidence and downstream DeepSeek extraction compared with sending the whole sheet to Tesseract.

## What is included

- Hespi 0.6.2 sheet-component detection.
- Primary specimen label selection.
- Hespi label-field detection.
- Tesseract OCR on detected field crops using page segmentation mode 6.
- Field names included in the DeepSeek prompt, while raw OCR text remains unchanged for evaluation.
- Per-record fallback to a primary-label crop or the full image.
- A 20-record A/B comparison against the existing full-image pipeline.

Hespi HTR, fuzzy reference matching, and Hespi's own LLM correction are disabled. The existing SCRIBE schema, DeepSeek official no-RAG extraction, reconciliation, and evaluation remain in control.

## Run locally

Hespi requires Python 3.10 or 3.11.

```bash
python -m pip install -e '.[dev,hespi]'
export DEEPSEEK_API_KEY="..."
python scripts/run_pipeline.py --config configs/hespi_lite_eval_20_baseline.yaml
python scripts/run_pipeline.py --config configs/hespi_lite_eval_20.yaml
python scripts/compare_hespi_lite_poc.py
```

The pretrained Hespi weights are downloaded on first use.

## Run in GitHub Actions

Open **Actions -> Hespi-lite POC -> Run workflow**. This workflow is manual-only and uses the repository secret `DEEPSEEK_API_KEY_SELF`.

The artifact contains:

- full-image baseline outputs;
- Hespi component and field manifests;
- annotated detector previews and field crops;
- per-region OCR;
- raw DeepSeek responses and diagnostics;
- field-level evaluation and a combined comparison report.

The main report is `reports/hespi_lite_poc/hespi_lite_poc_comparison.md`.

## Decision rule

Continue with Hespi only if it improves OCR evidence proxy and DeepSeek field metrics on the same EVAL records, while keeping fallback rates acceptable. This 20-record POC is too small for a production claim, and possible similarity between Hespi's pretraining data and the Zenodo benchmark must be investigated before generalising the result.
