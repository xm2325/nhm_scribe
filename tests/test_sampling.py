import pandas as pd
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
