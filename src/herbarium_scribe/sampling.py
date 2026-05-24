from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def stratified_random_sample(df: pd.DataFrame, n: int, by: str, seed: int = 42) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    n = min(n, len(df))
    if by not in df.columns or df[by].nunique(dropna=False) <= 1:
        return df.sample(n=n, random_state=seed).copy()
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(by, dropna=False))
    sizes = np.array([len(g) for _, g in groups], dtype=float)
    raw = sizes / sizes.sum() * n
    alloc = np.floor(raw).astype(int)
    # Give at least one row to non-empty strata while capacity allows.
    for i, (_, g) in enumerate(groups):
        if alloc.sum() < n and alloc[i] == 0 and len(g) > 0:
            alloc[i] = 1
    while alloc.sum() > n:
        candidates = np.where(alloc > 0)[0]
        j = candidates[np.argmin(raw[candidates] - np.floor(raw[candidates]))]
        alloc[j] -= 1
    while alloc.sum() < n:
        capacity = np.array([len(g) for _, g in groups]) - alloc
        candidates = np.where(capacity > 0)[0]
        if len(candidates) == 0:
            break
        # Largest fractional remainder, stable with seeded tie break.
        remainders = raw[candidates] - np.floor(raw[candidates])
        max_rem = remainders.max()
        tied = candidates[np.where(remainders == max_rem)[0]]
        j = int(rng.choice(tied))
        alloc[j] += 1
    parts = []
    for (name, g), k in zip(groups, alloc):
        if k > 0:
            parts.append(g.sample(n=min(k, len(g)), random_state=seed + abs(hash(str(name))) % 10000))
    out = pd.concat(parts, ignore_index=False) if parts else df.iloc[0:0]
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.sample(frac=1, random_state=seed).copy()


def make_demo_eval_split(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scfg = cfg.get("sampling", {})
    demo_n = int(scfg.get("demo_size", 2))
    eval_n = int(scfg.get("eval_size", 10))
    by = scfg.get("stratify_by", "institutionCode")
    seed = int(cfg.get("project", {}).get("random_state", 42))
    prefer_disjoint = bool(scfg.get("prefer_institution_disjoint", True))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    demo = None
    eval_df = None
    split_mode = "record_disjoint"

    if prefer_disjoint and by in df.columns and df[by].nunique() >= 2:
        institutions = list(df[by].drop_duplicates())
        rng = np.random.default_rng(seed)
        rng.shuffle(institutions)
        for inst in institutions:
            demo_pool = df[df[by] == inst]
            eval_pool = df[df[by] != inst]
            if len(demo_pool) >= min(demo_n, len(demo_pool)) and len(eval_pool) >= min(eval_n, len(eval_pool)) and min(demo_n, len(demo_pool)) > 0:
                demo = stratified_random_sample(demo_pool, min(demo_n, len(demo_pool)), by, seed)
                eval_df = stratified_random_sample(eval_pool, min(eval_n, len(eval_pool)), by, seed + 1)
                split_mode = "institution_disjoint"
                break
    if demo is None or eval_df is None:
        demo = stratified_random_sample(df, min(demo_n, len(df)), by, seed)
        remaining = df[~df["occurrenceID"].isin(set(demo["occurrenceID"]))]
        eval_df = stratified_random_sample(remaining, min(eval_n, len(remaining)), by, seed + 1)

    overlap = set(demo["occurrenceID"]) & set(eval_df["occurrenceID"])
    if overlap:
        raise ValueError(f"DEMO_SET and EVAL_SET overlap: {sorted(overlap)[:5]}")

    summary_rows = []
    for split_name, sub in [("DEMO_SET", demo), ("EVAL_SET", eval_df)]:
        counts = sub[by].value_counts(dropna=False).to_dict() if by in sub.columns else {"all": len(sub)}
        for key, count in counts.items():
            summary_rows.append({"split": split_name, "stratify_by": by, "stratum": key, "n": int(count), "split_mode": split_mode})
    return demo.reset_index(drop=True), eval_df.reset_index(drop=True), pd.DataFrame(summary_rows)


def save_split_outputs(demo: pd.DataFrame, eval_df: pd.DataFrame, summary: pd.DataFrame, processed: Path) -> None:
    demo.to_csv(processed / "demo_set.csv", index=False)
    eval_df.to_csv(processed / "eval_set.csv", index=False)
    summary.to_csv(processed / "split_summary.csv", index=False)
