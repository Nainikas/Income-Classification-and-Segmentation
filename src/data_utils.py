# tiny helpers to load this CPS dataset
# NOTE: the label column is not the usual <=50K/>50K. in this dump it's
# '- 50000.' and '50000+.' (sometimes with punctuation/spaces). normalize it.

from pathlib import Path
from typing import List, Tuple
import re
import pandas as pd

def load_columns(cols_path: str | Path) -> List[str]:
    return [line.strip() for line in Path(cols_path).read_text().splitlines() if line.strip()]

def load_census(data_path: str | Path, cols_path: str | Path) -> pd.DataFrame:
    cols = load_columns(cols_path)
    df = pd.read_csv(data_path, names=cols, header=None)
    # trim whitespace on strings (the file is pretty messy)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def _normalize_label(val: str) -> str | None:
    """map various income strings to '<=50K' or '>50K'."""
    if val is None:
        return None
    s = str(val).replace(" ", "").replace(".", "").replace(",", "").lower()
    s = s.replace("k", "000")  # 50k → 50000

    # direct forms seen in this file
    if s in {"<=50k", "<=50000", "<50k", "<50000", "-50000", "-50000$"}:
        return "<=50K"
    if s in {">50k", ">50000", "50000+", "50k+", "50000plus"}:
        return ">50K"

    # generic fallbacks
    if s.startswith(("<=", "<", "-")):
        return "<=50K"
    if s.endswith("+") or s.startswith(">"):
        return ">50K"

    m = re.fullmatch(r"-?\$?(\d+)", s)
    if m:
        return ">50K" if int(m.group(1)) > 50000 else "<=50K"

    return None  # if we get here, it's an unexpected form

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    assert "label" in df.columns, "expected 'label'"
    assert "weight" in df.columns, "expected 'weight'"

    y_fast = df["label"].str.strip().map({">50K": 1, "<=50K": 0})
    if y_fast.isna().any():
        y = df["label"].apply(_normalize_label).map({">50K": 1, "<=50K": 0})
        if y.isna().any():
            bad = df.loc[y.isna(), "label"].dropna().unique()[:10]
            raise ValueError(f"could not map some labels: {bad!r}")
    else:
        y = y_fast

    w = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(float)
    X = df.drop(columns=["label", "weight"])
    return X, y.astype(int), w
