import pandas as pd

from herbarium_scribe.evaluate import exact_match, token_f1, evidence_proxy, truthy_flag, assign_ocr_tertiles
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
