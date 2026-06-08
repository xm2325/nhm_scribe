import pandas as pd

from scripts.compare_hespi_v4_repeat10 import build_scale_gates, consensus_row


def test_consensus_auto_accepts_unanimous_direct_prediction():
    group = pd.DataFrame([
        {
            "repeat": f"repeat_{index}",
            "occurrenceID": "eval:1",
            "field": "typeStatus",
            "prediction": "ISOTYPE",
            "normalised_prediction": "isotype",
            "gold": "ISOTYPE",
            "normalised_gold": "isotype",
            "direct_evidence_supported": 1,
            "review_required": 0,
            "prediction_confidence": 0.95,
            "exact_match": 1,
            "token_f1": 1,
        }
        for index in (1, 2, 3)
    ])

    out = consensus_row(group)

    assert out["unanimous"] is True
    assert out["pairwise_agreement"] == 1
    assert out["consensus_action"] == "auto_accept"


def test_consensus_routes_non_allowlisted_field_to_review():
    group = pd.DataFrame([
        {
            "repeat": f"repeat_{index}",
            "occurrenceID": "eval:1",
            "field": "scientificName",
            "prediction": "Carex nigra",
            "normalised_prediction": "carex nigra",
            "gold": "Carex nigra",
            "normalised_gold": "carex nigra",
            "direct_evidence_supported": 1,
            "review_required": 0,
            "prediction_confidence": 0.99,
            "exact_match": 1,
            "token_f1": 1,
        }
        for index in (1, 2, 3)
    ])

    out = consensus_row(group)

    assert out["consensus_action"] == "human_review"
    assert out["auto_accept_policy_reason"] == "field_requires_human_review"


def test_catalog_auto_accept_requires_targeted_alphanumeric_evidence():
    base = {
        "occurrenceID": "eval:1",
        "field": "catalogNumber",
        "gold": "K0001",
        "normalised_gold": "K0001",
        "direct_evidence_supported": 1,
        "review_required": 0,
        "prediction_confidence": 0.95,
        "exact_match": 1,
        "token_f1": 1,
    }
    group = pd.DataFrame([
        {
            **base,
            "repeat": f"repeat_{index}",
            "prediction": "K0001",
            "normalised_prediction": "K0001",
        }
        for index in (1, 2, 3)
    ])
    targeted_ocr = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "region_type": "number",
        "ocr_text": "K 0001",
    }])

    assert consensus_row(group, targeted_ocr)["consensus_action"] == "auto_accept"

    whole_sheet_only = targeted_ocr.assign(region_type="whole_sheet")
    out = consensus_row(group, whole_sheet_only)
    assert out["consensus_action"] == "human_review"
    assert out["auto_accept_policy_reason"] == "catalog_requires_targeted_region_support"

    numeric_group = group.assign(prediction="202200", normalised_prediction="202200")
    out = consensus_row(numeric_group, targeted_ocr.assign(ocr_text="202200"))
    assert out["consensus_action"] == "human_review"
    assert out["auto_accept_policy_reason"] == "catalog_requires_letters_and_digits"


def test_consensus_routes_unstable_prediction_to_review():
    values = ["China", "", "India"]
    group = pd.DataFrame([
        {
            "repeat": f"repeat_{index}",
            "occurrenceID": "eval:1",
            "field": "country",
            "prediction": value,
            "normalised_prediction": value.lower(),
            "gold": "China",
            "normalised_gold": "china",
            "direct_evidence_supported": 0,
            "review_required": int(bool(value)),
            "prediction_confidence": 0.6 if value else 0,
            "exact_match": int(value == "China"),
            "token_f1": int(value == "China"),
        }
        for index, value in enumerate(values, start=1)
    ])

    out = consensus_row(group)

    assert out["unanimous"] is False
    assert out["pairwise_agreement"] == 0
    assert out["consensus_action"] == "human_review"
    assert out["consensus_review_priority"] == "high"


def test_scale_gate_rejects_stable_but_inaccurate_run():
    metrics = pd.DataFrame([{"parse_success_rate": 1.0}])
    overall = {
        "pairwise_agreement_rate": 0.98,
        "mean_unsupported_prediction_rate": 0.10,
        "human_review_rate": 0.20,
        "consensus_exact_match": 0.12,
        "consensus_token_f1": 0.15,
        "auto_accept_exact_match": 1.0,
    }

    gates = build_scale_gates(metrics, overall)

    assert not gates["passed"].all()
    assert not gates.loc[gates["gate"].eq("consensus_exact_match"), "passed"].item()
