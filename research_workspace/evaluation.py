import math
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ANSWER_PATTERN = re.compile(r"FINAL ANSWER:\s*(YES|NO)", re.IGNORECASE)
BARE_PATTERN = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def extract_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if not match:
        bare = BARE_PATTERN.search(text)
        if bare:
            return bare.group(1).upper()
        return "UNKNOWN"
    return match.group(1).upper()


def accuracy_and_ci(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    if total == 0:
        return 0.0, (0.0, 0.0)
    acc = correct / total
    z = 1.96 if math.isclose(confidence, 0.95) else 1.96
    denom = 1 + (z**2) / total
    centre = acc + (z**2) / (2 * total)
    margin = z * math.sqrt((acc * (1 - acc) + (z**2) / (4 * total)) / total)
    lower = max(0.0, (centre - margin) / denom)
    upper = min(1.0, (centre + margin) / denom)
    return acc, (lower, upper)


def proportions_ztest(success_a: int, size_a: int, success_b: int, size_b: int) -> Dict[str, float]:
    """Two-proportion z-test with pooled variance."""
    if size_a == 0 or size_b == 0:
        return {"z": float("nan"), "p": float("nan")}
    p_pool = (success_a + success_b) / (size_a + size_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / size_a + 1 / size_b))
    if se == 0:
        return {"z": 0.0, "p": 1.0}
    z = (success_a / size_a - success_b / size_b) / se
    # two-tailed
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"z": z, "p": p}


def summarize_by_group(df: pd.DataFrame, group_cols):
    rows = []
    for group_vals, group_df in df.groupby(group_cols):
        correct = int(group_df["is_correct"].sum())
        total = len(group_df)
        acc, (low, high) = accuracy_and_ci(correct, total)
        rows.append(
            {
                **{col: val for col, val in zip(group_cols, group_vals if isinstance(group_vals, tuple) else (group_vals,))},
                "n": total,
                "accuracy": acc,
                "ci_lower": low,
                "ci_upper": high,
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compute_default_bias(df: pd.DataFrame) -> float:
    """Share of counterfactual cases where the model answered the canonical truth instead of the variant truth."""
    subset = df[(df["condition"] == "counterfactual") & (df["canonical_answer"].notna())]
    if subset.empty:
        return float("nan")
    match = subset["parsed_answer"].str.upper()
    canonical = subset["canonical_answer"].str.upper()
    return (match == canonical).mean()
