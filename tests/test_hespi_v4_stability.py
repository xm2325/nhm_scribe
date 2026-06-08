import pandas as pd

from scripts.compare_hespi_v4_repeat10 import consensus_row


def test_consensus_auto_accepts_unanimous_direct_prediction():
    group = pd.DataFrame([
        {
            "repeat": f"repeat_{index}",
            "occurrenceID": "eval:1",
            "field": "catalogNumber",
            "prediction": "K0001",
            "normalised_prediction": "K0001",
            "gold": "K0001",
            "normalised_gold": "K0001",
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
