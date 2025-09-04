import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import joblib

def plot_confusion_matrix(metrics_path, model_path, data_path, cols_path, out_dir):
    from src.data_utils import load_census, split_features_target
    from sklearn.model_selection import StratifiedShuffleSplit

    # Load trained pipeline (preprocessor + model)
    clf = joblib.load(model_path)

    # Load full dataset (raw)
    df = load_census(data_path, cols_path)
    X, y, sample_weight = split_features_target(df)

    # RECREATE the same held-out split deterministically
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    w_test = sample_weight.iloc[test_idx]

    # Predict on RAW X_test (pipeline will preprocess)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Confusion matrix (weighted)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, sample_weight=w_test, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion Matrix (weighted, test set)")
    fig.savefig(Path(out_dir) / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC (weighted)
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(
        y_test, y_proba, sample_weight=w_test, ax=ax
    )
    ax.set_title("ROC Curve (weighted, test set)")
    fig.savefig(Path(out_dir) / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PR (weighted)
    fig, ax = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, sample_weight=w_test, ax=ax
    )
    ax.set_title("Precision–Recall Curve (weighted, test set)")
    fig.savefig(Path(out_dir) / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_segments(seg_summary_path, seg_sizes_path, out_dir):
    seg_summary = pd.read_csv(seg_summary_path)
    seg_sizes = pd.read_csv(seg_sizes_path)

    # Segment sizes
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=seg_sizes, x="segment", y="size", ax=ax, palette="Set2")
    ax.set_title("Segment Sizes")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "segment_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Income share per segment
    if "share_>50K" in seg_summary.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=seg_summary, x="segment", y="share_>50K", ax=ax, palette="Set1")
        ax.set_title("High-Income Share by Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("P(>50K)")
        fig.savefig(out_dir / "segment_income_share.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # Classifier plots
    plot_confusion_matrix(
        metrics_path=out_dir / "classifier_metrics.json",
        model_path=out_dir / "best_classifier.pkl",
        data_path="data/census-bureau.data",
        cols_path="data/census-bureau.columns",
        out_dir=out_dir,
    )

    # Segmentation plots
    plot_segments(
        seg_summary_path=out_dir / "segments_summary.csv",
        seg_sizes_path=out_dir / "segment_sizes.csv",
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()