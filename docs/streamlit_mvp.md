# Streamlit MVP

This repository includes a first-pass Streamlit app for exploring the Herbarium SCRIBE pipeline.

## Run Locally

```bash
python -m pip install -e .
python -m pip install streamlit
streamlit run streamlit_app.py
```

For the live DeepSeek demo, set:

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

## Streamlit Community Cloud

Use:

```text
Main file path: streamlit_app.py
```

Add these app secrets:

```toml
DEEPSEEK_API_KEY = "..."
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"
```

`packages.txt` installs the Tesseract binary for the live OCR tab.

## Included Demo Data

The app ships with `app_data/real_eval_100_streamlit_bundle.zip`, a sanitised bundle built from the successful GitHub Actions eval100 artifact. It contains the 100-record evaluation CSVs, OCR outputs, LLM outputs, RAG contexts, diagnostics, and report markdown needed by the app. It removes provider response bodies and reasoning traces before committing the bundle.
