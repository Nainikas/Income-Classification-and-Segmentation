# k-means segmentation with a simple silhouette sweep. writes segment summaries.
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .data_utils import load_census, _normalize_label
from .preprocess import build_preprocessor
from .report_utils import save_df_csv, save_df_parquet, save_json, write_text

def _profile(df: pd.DataFrame, seg_col: str) -> pd.DataFrame:
    rows = []
    for seg, g in df.groupby(seg_col):
        row = {"segment": int(seg), "size": int(len(g))}
        for c in df.columns:
            if c in [seg_col, "weight", "label_norm"]:
                continue
            if df[c].dtype == "object":
                m = g[c].mode(dropna=True)
                row[c] = m.iloc[0] if not m.empty else None
            else:
                row[c] = float(g[c].median())
        if "label_norm" in df.columns:
            row["share_>50K"] = float((g["label_norm"] == ">50K").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("segment")

def _personas(summary: pd.DataFrame) -> str:
    lines = ["# Segment Personas (draft)\n"]
    for _, r in summary.sort_values("segment").iterrows():
        hdr = f"## Segment {int(r['segment'])} (n={int(r['size'])})"
        if "share_>50K" in r and pd.notna(r["share_>50K"]):
            hdr += f" - P(>50K) ~ {r['share_>50K']:.2f}"
        lines.append(hdr)

        picks = []
        for col, label in [
            ("education", "Education"),
            ("class of worker", "Worker class"),
            ("major industry code", "Industry"),
            ("marital stat", "Marital"),
            ("sex", "Sex"),
        ]:
            if col in summary.columns and pd.notna(r.get(col)):
                picks.append(f"{label}: {r[col]}")
        lines.append("- " + "; ".join(picks) if picks else "- mixed attributes")
        lines.append("")
    return "\n".join(lines)

def main(a):
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_census(a.data_path, a.cols_path)

    if "label" in df.columns:
        df["label_norm"] = df["label"].apply(_normalize_label)

    # features only (drop label, weight)
    drop_cols = [c for c in ["label", "weight", "label_norm"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    pre = build_preprocessor(X)
    X_proc = pre.fit_transform(X)

    # pick k via silhouette on a subsample
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_proc.shape[0], size=min(8000, X_proc.shape[0]), replace=False)
    best_k, best_s = None, -1.0
    for k in range(a.k_min, a.k_max + 1):
        lbl = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_proc[sample_idx])
        s = silhouette_score(X_proc[sample_idx], lbl)
        if s > best_s:
            best_s, best_k = s, k

    final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    seg = final.fit_predict(X_proc)

    df_out = df.copy()
    df_out["segment"] = seg

    summary = _profile(df_out, "segment")
    if "label_norm" in df_out.columns:
        summary["label_present"] = True

    save_df_parquet(df_out, out_dir / "segments.parquet")
    save_df_csv(summary, out_dir / "segments_summary.csv")
    sizes = pd.DataFrame({"segment": np.arange(best_k), "size": np.bincount(seg)})
    save_df_csv(sizes, out_dir / "segment_sizes.csv")
    save_json({"best_k": best_k, "silhouette": float(best_s)}, out_dir / "segmentation_report.json")

    write_text(_personas(summary), out_dir / "segment_personas.md")
    print(json.dumps({"best_k": best_k, "silhouette": float(best_s)}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--cols_path", required=True)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--k_min", default=3, type=int)
    p.add_argument("--k_max", default=8, type=int)
    args = p.parse_args()
    main(args)
