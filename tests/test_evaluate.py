import pandas as pd

from herbarium_scribe.evaluate import (
    assign_ocr_tertiles,
    evaluate_predictions,
    evidence_proxy,
    evidence_support_status,
    exact_match,
    field_exact_match,
    field_token_f1,
    normalize_field_value,
    review_decision,
    token_f1,
    truthy_flag,
)
from herbarium_scribe.report import _rag_delta_tables


def test_exact_match_normalises_case():
    assert exact_match("Rosa canina", "rosa canina") == 1


def test_token_f1_partial_overlap():
    assert 0 < token_f1("Rosa", "Rosa canina") < 1


def test_evidence_proxy_is_not_cer():
    assert evidence_proxy("Rosa canina", "label text Rosa canina here") == 1.0


def test_truthy_flag_handles_csv_false_strings():
    assert truthy_flag("False") is False
    assert truthy_flag("0") is False
    assert truthy_flag("True") is True


def test_assign_ocr_tertiles_handles_duplicate_quantile_edges():
    scores = pd.Series([0, 0, 0, 0, 0, 0.2, 0.3, 0.4, 0.5, 1.0])
    tertiles = assign_ocr_tertiles(scores)
    assert len(tertiles) == len(scores)
    assert set(tertiles).issubset({"low", "medium", "high"})


def test_field_normalizers_handle_common_herbarium_format_variants():
    assert normalize_field_value("catalogNumber", "B 10 0355192") == "B100355192"
    assert field_exact_match("catalogNumber", "TU 351001", "TU351001") == 1
    assert field_exact_match("eventDate", "19 Aug 1960", "1960-08-19") == 1
    assert field_exact_match("recordedBy", "K. I. Lahtivirta", "Lahtivirta, K. I.") == 1
    assert field_exact_match("typeStatus", "Isotype of Carex lutensis", "isotype") == 1
    assert field_exact_match("decimalLatitude", "51.50004", "51.5") == 1
    assert field_token_f1("eventDate", "1953-12-05/1953-12-07", "1953-12-05") > 0


def test_evaluation_reports_direct_ocr_evidence_support():
    pred = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "method": "deepseek",
        "catalogNumber": "K0001",
        "catalogNumber_evidence_span": "K0001",
        "catalogNumber_confidence": 0.9,
        "country": "Cuba",
        "country_evidence_span": "Cuba",
        "country_confidence": 0.7,
    }])
    gold = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "catalogNumber": "K0001",
        "country": "Netherlands",
    }])
    ocr = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "ocr_text": "Herbarium label K0001 Netherlands",
    }])

    detail, summary, _ = evaluate_predictions(
        pred,
        gold,
        ocr,
        ["catalogNumber", "country"],
    )

    catalog = detail[detail["field"] == "catalogNumber"].iloc[0]
    country = detail[detail["field"] == "country"].iloc[0]
    assert catalog["direct_evidence_supported"] == 1
    assert catalog["evidence_support_status"] == "direct"
    assert catalog["review_required"] == 0
    assert country["unsupported_prediction"] == 1
    assert country["review_required"] == 1
    assert country["review_priority"] == "high"
    assert summary["unsupported_prediction_rate"].mean() == 0.5


def test_evidence_status_and_review_gate_distinguish_contextual_inference():
    status = evidence_support_status(
        True,
        "Date? (4.1960",
        1.0,
        0.0,
        0.0,
    )
    required, priority, reasons = review_decision(
        field="eventDate",
        predicted=True,
        support_status=status,
        confidence=0.8,
        validation_warning=False,
    )
    assert status == "contextual_inference"
    assert required is True
    assert priority == "high"
    assert "high_risk_without_direct_evidence" in reasons


def test_catalog_with_multiple_decoded_barcodes_requires_review(tmp_path):
    predictions = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "method": "llm",
        "catalogNumber": "L.2489324",
        "catalogNumber_evidence_span": "L.2489324",
        "catalogNumber_confidence": 0.95,
        "parse_failure": False,
        "not_evaluated": False,
    }])
    gold = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "catalogNumber": "L.2489324",
    }])
    ocr = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "ocr_engine": "zxingcpp",
        "ocr_text": "L.2489324\nL 0398679",
    }])

    detail, _, _ = evaluate_predictions(
        predictions,
        gold,
        ocr,
        ["catalogNumber"],
        {"processed": tmp_path},
    )

    row = detail.iloc[0]
    assert row["review_required"] == 1
    assert row["review_priority"] == "high"
    assert "multiple_decoded_barcodes" in row["review_reasons"]
    assert row["barcode_candidate_count"] == 2


def test_catalog_mismatch_with_single_decoded_barcode_requires_review(tmp_path):
    predictions = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "method": "llm",
        "catalogNumber": "E00120034",
        "catalogNumber_evidence_span": "E00120034",
        "catalogNumber_confidence": 0.95,
        "parse_failure": False,
        "not_evaluated": False,
    }])
    gold = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "catalogNumber": "E00120084",
    }])
    ocr = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "ocr_engine": "zxingcpp",
        "ocr_text": "E00120084",
    }])

    detail, _, _ = evaluate_predictions(
        predictions,
        gold,
        ocr,
        ["catalogNumber"],
        {"processed": tmp_path},
    )

    row = detail.iloc[0]
    assert row["review_required"] == 1
    assert "catalog_mismatch_single_decoded_barcode" in row["review_reasons"]


def test_rag_delta_reports_no_rag_only(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame([
        {
            "occurrenceID": "eval:1",
            "method": "deepseek_v4_pro_no_rag",
            "field": "catalogNumber",
            "exact_match": "",
            "token_f1": "",
            "evaluable": 0,
            "not_evaluated": 1,
            "parse_failure": 0,
        }
    ]).to_csv(processed / "real_eval_1_evaluation_detail.csv", index=False)
    helped, hurt, verdict = _rag_delta_tables(
        {"processed": processed},
        {"method_name": "deepseek_v4_pro_no_rag", "outputs": {"prefix": "real_eval_1"}},
    )
    assert len(helped) == 0
    assert len(hurt) == 0
    assert "only no-RAG" in verdict
