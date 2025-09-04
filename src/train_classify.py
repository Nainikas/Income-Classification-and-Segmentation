# train + evaluate a binary income classifier; writes metrics + pipeline pickle
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from .data_utils import load_census, split_features_target
from .preprocess import build_preprocessor
from .report_utils import save_json, save_df_csv

def _fit_eval(name, model, prep, X_tr, y_tr, w_tr, X_te, y_te, w_te, out_dir: Path):
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)

    p = pipe.predict_proba(X_te)[:, 1]
    yhat = (p >= 0.5).astype(int)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_te, yhat, sample_weight=w_te)),
        "roc_auc": float(roc_auc_score(y_te, p, sample_weight=w_te)),
        "pr_auc": float(average_precision_score(y_te, p, sample_weight=w_te)),
    }
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, yhat, average="binary", sample_weight=w_te, zero_division=0)
    metrics.update({"precision": float(pr), "recall": float(rc), "f1": float(f1)})
    cm = confusion_matrix(y_te, yhat, sample_weight=w_te).astype(float)
    metrics["confusion_matrix"] = cm.tolist()

    # rough permutation importance (feature blocks). it's coarse but fine for the report.
    try:
        idx = np.random.RandomState(42).choice(len(X_te), size=min(5000, len(X_te)), replace=False)
        pi = permutation_importance(pipe, X_te.iloc[idx], y_te.iloc[idx], scoring="roc_auc", n_repeats=5, random_state=42)
        base_cols = list(X_tr.columns)
        imp = pd.DataFrame({
            "feature_block": base_cols,
            "import_mean": pi.importances_mean[:len(base_cols)],
            "import_std":  pi.importances_std[:len(base_cols)],
        }).sort_values("import_mean", ascending=False)
    except Exception:
        imp = pd.DataFrame({"feature_block": [], "import_mean": [], "import_std": []})

    save_df_csv(imp, out_dir / f"feature_importance_{name}.csv")
    return pipe, metrics

def main(a):
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_census(a.data_path, a.cols_path)
    X, y, w = split_features_target(df)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(X, y, w, test_size=0.2, stratify=y, random_state=42)
    prep = build_preprocessor(X)

    # baseline + tree
    candidates = []

    lr = LogisticRegression(solver="saga", penalty="l2", max_iter=5000, n_jobs=-1)
    pipe_lr, m_lr = _fit_eval("logreg", lr, prep, X_tr, y_tr, w_tr, X_te, y_te, w_te, out_dir)
    candidates.append(("logreg", pipe_lr, m_lr))

    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=400, learning_rate=0.05,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        pipe_lgbm, m_lgbm = _fit_eval("lightgbm", lgbm, prep, X_tr, y_tr, w_tr, X_te, y_te, w_te, out_dir)
        candidates.append(("lightgbm", pipe_lgbm, m_lgbm))
    except Exception:
        pass  # if lightgbm isn’t installed, we still have a baseline

    best_name, best_pipe, best_metrics = sorted(candidates, key=lambda x: x[2]["roc_auc"], reverse=True)[0]
    save_json(best_metrics, out_dir / "classifier_metrics.json")
    dump(best_pipe, out_dir / "best_classifier.pkl")
    print(json.dumps(best_metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--cols_path", required=True)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    main(args)
