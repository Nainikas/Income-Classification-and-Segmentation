from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cats = X.select_dtypes(include=["object"]).columns.tolist()
    nums = X.select_dtypes(exclude=["object"]).columns.tolist()
    return cats, nums

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # one-hot categoricals, standardize numerics. simple on purpose.
    cats, nums = _infer_columns(X)
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cats),
            ("num", StandardScaler(), nums),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
