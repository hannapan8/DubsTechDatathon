"""
Clustering model for AccessGuru website-level features.

Expected input:
  dubstechdatathon/data/website_ml_features.csv

Outputs:
  dubstechdatathon/ml-model/outputs/clusters.csv
  dubstechdatathon/ml-model/outputs/elbow.png
  dubstechdatathon/ml-model/outputs/pca_clusters.png
  dubstechdatathon/ml-model/outputs/cluster_summary.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing, errors="ignore")


# Load website-level features
def get_feature_matrix(
    df: pd.DataFrame,
    id_cols: list[str],
    label_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      meta_df: id + label columns (kept for joining results)
      X_df: numeric feature columns for ML
    """
    meta_cols = [c for c in (id_cols + label_cols) if c in df.columns]
    meta_df = df[meta_cols].copy()

    X_df = df.drop(columns=meta_cols, errors="ignore").copy()
    X_df = X_df.dropna(axis=1, how="all")
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df = X_df.fillna(0)
    nunique = X_df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X_df = X_df.drop(columns=const_cols)

    return meta_df, X_df

# plotting the clausters in a 2D scatterplot
def plot_elbow(X_scaled: np.ndarray, outpath: Path, kmin: int = 2, kmax: int = 10) -> None:
    inertias = []
    ks = list(range(kmin, kmax + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for KMeans")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pca_scatter(df_out: pd.DataFrame, outpath: Path, color_col: str = "cluster") -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df_out["pca1"], df_out["pca2"], c=df_out[color_col])
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Website Clusters (PCA)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train clustering model on website-level accessibility features.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "website_ml_features.csv"),
        help="Path to website_ml_features.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory to write outputs",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of clusters for KMeans")
    parser.add_argument("--make_elbow", action="store_true", help="If set, generate elbow plot (k=2..10)")
    parser.add_argument("--pca_components", type=int, default=2, help="PCA components to compute (2 recommended)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    print("Raw shape:", df.shape)
    if "nan" in df.columns:
        df = df.drop(columns=["nan"])
        print('Dropped column "nan"')

    # Identify metadata columns
    id_cols = ["web_URL_id", "web_URL", "id"]
    label_cols = ["domain_category"]

    meta_df, X_df = get_feature_matrix(df, id_cols=id_cols, label_cols=label_cols)

    print("Meta columns:", list(meta_df.columns))
    print("Feature matrix shape:", X_df.shape)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    # Optional elbow plot
    if args.make_elbow:
        elbow_path = output_dir / "elbow.png"
        print(f"Saving elbow plot: {elbow_path}")
        plot_elbow(X_scaled, elbow_path, kmin=2, kmax=10)

    # KMeans clustering
    print(f"Training KMeans with k={args.k}")
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    # Silhouette score
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"Silhouette score (k={args.k}): {sil_score:.3f}")

    # PCA for visualization
    if args.pca_components >= 2:
        pca = PCA(n_components=args.pca_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca1, pca2 = X_pca[:, 0], X_pca[:, 1]
    else:
        # Still create columns
        pca1 = np.zeros(len(X_scaled))
        pca2 = np.zeros(len(X_scaled))

    # Build output dataframe (meta + cluster + PCA)
    df_out = meta_df.copy()
    df_out["cluster"] = clusters
    df_out["pca1"] = pca1
    df_out["pca2"] = pca2

    # Add a couple interpretable per-cluster stats
    df_out["total_violations"] = df.get("violation_count_sum", pd.Series([np.nan] * len(df))).values
    df_out["avg_score"] = df.get("violation_score_mean", pd.Series([np.nan] * len(df))).values

    # Save cluster assignments
    clusters_path = output_dir / "clusters.csv"
    print(f"Saving clusters: {clusters_path}")
    df_out.to_csv(clusters_path, index=False)

    # Cluster summary table
    summary_cols = []
    for c in ["domain_category", "total_violations", "avg_score"]:
        if c in df_out.columns:
            summary_cols.append(c)

    if summary_cols:
        summary = (
            df_out.groupby("cluster")[summary_cols]
            .agg(
                n_websites=("domain_category", "count") if "domain_category" in summary_cols else ("pca1", "count"),
                domains=("domain_category", lambda s: ", ".join(sorted(set(map(str, s)))) ) if "domain_category" in summary_cols else ("pca1", lambda _: ""),
                avg_total_violations=("total_violations", "mean") if "total_violations" in summary_cols else ("pca1", "mean"),
                avg_score=("avg_score", "mean") if "avg_score" in summary_cols else ("pca1", "mean"),
            )
            .reset_index()
        )
        summary_path = output_dir / "cluster_summary.csv"
        print(f"Saving cluster summary: {summary_path}")
        summary.to_csv(summary_path, index=False)

    # Save PCA plot
    pca_plot_path = output_dir / "pca_clusters.png"
    print(f"Saving PCA plot: {pca_plot_path}")
    plot_pca_scatter(df_out, pca_plot_path, color_col="cluster")

    print("\nDone.")
    print("Outputs written to:", output_dir)


if __name__ == "__main__":
    main()
