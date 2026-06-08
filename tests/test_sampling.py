import pandas as pd
from herbarium_scribe.pipeline import stage_metadata
from herbarium_scribe.sampling import make_demo_eval_split


def test_demo_eval_split_has_no_overlap():
    df = pd.DataFrame({"occurrenceID": [f"id{i}" for i in range(12)], "institutionCode": ["A"]*4 + ["B"]*4 + ["C"]*4})
    cfg = {"project": {"random_state": 42}, "sampling": {"demo_size": 2, "eval_size": 6, "stratify_by": "institutionCode"}}
    demo, eval_df, summary = make_demo_eval_split(df, cfg)
    assert set(demo["occurrenceID"]).isdisjoint(set(eval_df["occurrenceID"]))
    assert len(summary) > 0


def test_demo_eval_split_is_stable_with_same_seed():
    df = pd.DataFrame({"occurrenceID": [f"id{i}" for i in range(12)], "institutionCode": ["A"]*4 + ["B"]*4 + ["C"]*4})
    cfg = {"project": {"random_state": 42}, "sampling": {"demo_size": 2, "eval_size": 6, "stratify_by": "institutionCode"}}
    demo1, eval1, _ = make_demo_eval_split(df, cfg)
    demo2, eval2, _ = make_demo_eval_split(df, cfg)
    assert demo1["occurrenceID"].tolist() == demo2["occurrenceID"].tolist()
    assert eval1["occurrenceID"].tolist() == eval2["occurrenceID"].tolist()


def test_frozen_split_can_select_repeatable_stratified_calibration_subset(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    pd.DataFrame([
        {"occurrenceID": "demo:1", "institutionCode": "D", "catalogNumber": "D1"},
        {"occurrenceID": "demo:2", "institutionCode": "D", "catalogNumber": "D2"},
    ]).to_csv(frozen / "demo_set.csv", index=False)
    pd.DataFrame([
        {"occurrenceID": f"eval:{index}", "institutionCode": institution, "catalogNumber": f"E{index}"}
        for index, institution in enumerate(["A", "A", "B", "B", "C", "C"], start=1)
    ]).to_csv(frozen / "eval_set.csv", index=False)
    pd.DataFrame([{
        "split": "EVAL_SET",
        "stratify_by": "institutionCode",
        "stratum": "all",
        "n": 6,
        "split_mode": "frozen",
    }]).to_csv(frozen / "split_summary.csv", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  data_dir: {tmp_path / "run"}
  reports_dir: {tmp_path / "reports"}
sampling:
  frozen_split_dir: {frozen}
  frozen_eval_size: 3
  stratify_by: institutionCode
project:
  random_state: 42
""",
        encoding="utf-8",
    )

    _, first, first_summary = stage_metadata(config)
    _, second, second_summary = stage_metadata(config)

    assert len(first) == 3
    assert list(first["occurrenceID"]) == list(second["occurrenceID"])
    assert set(first["institutionCode"]) == {"A", "B", "C"}
    assert set(first_summary["split_mode"]) == {"frozen_stratified_subset"}
    assert first_summary.equals(second_summary)
