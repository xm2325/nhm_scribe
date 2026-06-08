import pandas as pd

from herbarium_scribe.evaluate import (
    assign_ocr_tertiles,
    evaluate_predictions,
    evidence_proxy,
    exact_match,
    field_exact_match,
    field_token_f1,
    normalize_field_value,
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
    assert country["unsupported_prediction"] == 1
    assert summary["unsupported_prediction_rate"].mean() == 0.5


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
