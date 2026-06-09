# Hespi v11 Qwen Primary-Label Vision

This experiment compares Tesseract with direct multimodal understanding on the
same automatically detected primary-label crops from the frozen ten-record EVAL
set.

## Pipeline

```text
shared validated specimen image
        |
        v
Hespi primary-label detection
        |
        +--> Tesseract transcription
        |
        +--> Qwen-VL image input
                |
                +--> complete line-preserving transcription
                +--> uncertain visual spans
                +--> structured herbarium fields with exact evidence
```

The Qwen output preserves the complete raw API response and complete visual
transcription before field extraction. A field is flagged as unsupported when
neither its value nor its evidence span appears in Qwen's own transcription.

## API configuration

The GitHub Actions workflow reads only:

```text
QWEN_API_KEY
```

The key is never committed or printed. The workspace endpoint is:

```text
https://ws-jta6lz3g3givn2ww.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
```

The model name is a workflow input and defaults to `qwen3.7-plus`. The workflow
queries the workspace `/models` endpoint and saves the advertised model IDs so
an unavailable deployment name can be diagnosed without exposing the key.

## Run

Use **Actions -> Hespi v11 eval10 Qwen primary-label vision -> Run workflow**.
Ordinary code pushes do not start the experiment. Updating
`.github/run-hespi-v11-qwen-vision.trigger` starts only this workflow.

## Outputs

The artifact includes:

- primary-label crops and bounding boxes;
- Tesseract text from each identical crop;
- complete Qwen raw response and model metadata;
- Qwen complete transcription and uncertain spans;
- direct multimodal field extraction;
- field-level exact match and token F1;
- gold-value evidence proxies for Qwen and Tesseract transcription;
- unsupported-prediction diagnostics;
- a Streamlit result bundle.

Do not describe the evidence proxies as OCR CER or WER. They only test whether a
gold metadata value appears in the transcription after normalization.
