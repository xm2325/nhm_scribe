from herbarium_scribe.pipeline import _parse_llm_json


def test_parse_llm_json_accepts_wrapped_fields():
    raw = """```json
    {"record": {"catalog_number": {"text": "E00633257", "confidence": 0.8}}}
    ```"""
    parsed = _parse_llm_json(raw)
    assert parsed["catalogNumber"]["value"] == "E00633257"
    assert parsed["catalogNumber"]["confidence"] == 0.8


def test_parse_llm_json_rejects_unrelated_json():
    assert _parse_llm_json('{"note": "not an extraction"}') is None
