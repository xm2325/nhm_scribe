from herbarium_scribe.schema import validate_record


def test_schema_validation_adds_missing_fields_and_clips_confidence():
    rec = {"catalogNumber": {"value": "E123", "confidence": 2, "evidence_span": "E123"}, "decimalLatitude": {"value": "91", "confidence": 0.5, "evidence_span": "91"}}
    out = validate_record(rec)
    assert out["catalogNumber"]["confidence"] == 1.0
    assert "decimalLatitude_out_of_range" in out["warnings"]
    assert "scientificName" in out
