import pandas as pd

from herbarium_scribe.pipeline import _parse_llm_json, stage_extract


def test_parse_llm_json_accepts_wrapped_fields():
    raw = """```json
    {"record": {"catalog_number": {"text": "E00633257", "confidence": 0.8}}}
    ```"""
    parsed = _parse_llm_json(raw)
    assert parsed["catalogNumber"]["value"] == "E00633257"
    assert parsed["catalogNumber"]["confidence"] == 0.8


def test_parse_llm_json_rejects_unrelated_json():
    assert _parse_llm_json('{"note": "not an extraction"}') is None


def test_empty_llm_response_is_not_evaluated(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  data_dir: {data_dir}
  reports_dir: {tmp_path / "reports"}
method_name: deepseek_v4_pro_no_rag
llm:
  backend: deepseek_api
rag:
  enabled: false
outputs:
  prefix: smoke
  prediction_name: deepseek
  llm_outputs_name: smoke_outputs.jsonl
evaluation:
  fields:
    - catalogNumber
""",
        encoding="utf-8",
    )
    pd.DataFrame([{"occurrenceID": "demo:1", "institutionCode": "D", "catalogNumber": "D1"}]).to_csv(processed / "demo_set.csv", index=False)
    pd.DataFrame([{"occurrenceID": "eval:1", "institutionCode": "E", "catalogNumber": "E1"}]).to_csv(processed / "eval_set.csv", index=False)
    pd.DataFrame([{"occurrenceID": "eval:1", "ocr_text": "E1", "region_id": "full"}]).to_csv(processed / "ocr_by_region.csv", index=False)

    def fake_llm(_messages, _cfg):
        return {
            "backend": "deepseek_api",
            "requested_model": "deepseek-v4-pro",
            "actual_model": "deepseek-v4-pro",
            "content": "",
            "error_message": "",
            "endpoint_reachable": True,
            "api_key_present": True,
            "base_url": "https://api.deepseek.com",
            "finish_reason": "stop",
            "message": {"content": ""},
            "message_keys": ["content"],
            "reasoning_content": "",
            "usage": {"total_tokens": 12},
            "response": {"model": "deepseek-v4-pro", "choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
        }

    monkeypatch.setattr("herbarium_scribe.pipeline.call_llm_with_metadata", fake_llm)
    out = stage_extract(config_path)
    llm = out[out["method"] == "deepseek_v4_pro_no_rag"].iloc[0]
    assert bool(llm["not_evaluated"]) is True
    assert llm["not_evaluated_reason"] == "empty_raw_output"

    raw = (data_dir / "interim" / "llm" / "smoke_outputs.jsonl").read_text(encoding="utf-8")
    assert '"response_body"' in raw
    assert '"response_finish_reason": "stop"' in raw
