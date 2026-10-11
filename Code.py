"""
Clinical ICU Patient Phenotyping & Sepsis Severity Profiling Pipeline
====================================================================
Dataset: Sepsis_data.csv
Framework: Scikit-Learn, SciPy, Pandas, NumPy, Matplotlib, Seaborn
Techniques:
  - Robust Imputation & Yeo-Johnson Power Transforms
  - Truncated SVD & PCA with Factor Loading Extraction
  - t-SNE 2D Manifold Non-linear Embeddings
  - Model Selection: K-Means (Silhouette/Davies-Bouldin) vs. GMM (BIC/AIC)
  - Density-Based Anomaly Scoring (Tail Probabilities)
  - Cluster-Stratified Lasso (L1) and Ridge (L2) Regularization
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import train_test_split

# Set aesthetic visualization style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_theme(style="whitegrid", palette="muted")


def load_and_preprocess(csv_path: str, id_col_hint: str = "Patient_ID", target_col_hint: str = "ICULOS"):
    """
    Loads Sepsis CSV data, handles longitudinal aggregation if needed,
    removes ultra-sparse features, and applies robust Yeo-Johnson transformations.
    """
    # Check for file existence (with or without .csv extension)
    if not os.path.exists(csv_path):
        if os.path.exists(csv_path + ".csv"):
            csv_path = csv_path + ".csv"
        else:
            raise FileNotFoundError(f"Cannot locate dataset at: {csv_path} or {csv_path}.csv")

    print(f"[*] Ingesting dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Raw Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Detect ID column
    id_col = None
    for cand in [id_col_hint, "Patient_ID", "patient_id", "id", "ID", "Unnamed: 0"]:
        if cand in df.columns:
            id_col = cand
            break

    # Longitudinal aggregation if multi-hourly records exist per patient
    if id_col and df[id_col].nunique() < len(df) and df[id_col].nunique() > 10:
        print(f"[*] Multi-hour records detected. Aggregating summary statistics by '{id_col}'...")
        num_cols = df.select_dtypes(include=[np.number]).columns.drop(id_col)
        df_agg = df.groupby(id_col)[num_cols].agg(["mean", "max", "min"]).reset_index()
        df_agg.columns = [f"{c}_{stat}" if stat else c for c, stat in df_agg.columns]
        df = df_agg

    # Detect Target Column (Length of Stay / Severity metric)
    target_col = None
    for cand in [target_col_hint, f"{target_col_hint}_max", "ICULOS", "ICULOS_max", "SepsisLabel", "SepsisLabel_max", "LOS"]:
        if cand in df.columns:
            target_col = cand
            break

    target = None
    if target_col:
        print(f"[*] Target variable identified: '{target_col}'")
        target = df[target_col].values
        features_df = df.drop(columns=[target_col])
    else:
        features_df = df.copy()

    if id_col and id_col in features_df.columns:
        features_df = features_df.drop(columns=[id_col])

    # Drop non-numeric / metadata columns
    metadata_cols = [c for c in ["time", "Hour", "Unit1", "Unit2", "Gender"] if c in features_df.columns]
    if metadata_cols:
        features_df = features_df.drop(columns=metadata_cols)

    # Retain numerical features with missingness < 80%
    numeric_df = features_df.select_dtypes(include=[np.number])
    valid_cols = numeric_df.columns[numeric_df.isnull().mean() < 0.80]
    X_raw = numeric_df[valid_cols].values
    feature_names = list(valid_cols)
    print(f"    Retained Clean Numeric Features: {len(feature_names)}")

    # 1. Median Imputation
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # 2. Power Transformation (Yeo-Johnson) for skewed lab markers
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    X_scaled = transformer.fit_transform(X_imputed)

    return X_scaled, target, feature_names


def run_dimensionality_reduction(X: np.ndarray, feature_names: list, variance_threshold: float = 0.85):
    """
    Computes PCA & Truncated SVD, extracts loadings, and plots Explained Variance + Feature Loadings.
    """
    print("\n" + "="*60)
    print("[*] STEP 1: Dimensionality Reduction (PCA & Truncated SVD)")
    print("="*60)

    # Full PCA Spectrum
    pca_full = PCA().fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
    print(f"    Components required for {variance_threshold*100:.0f}% cumulative variance: {n_components}/{X.shape[1]}")

    # Fitted PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Loadings Matrix (PC1 & PC2)
    loadings = pd.DataFrame(
        pca.components_[:2, :].T,
        index=feature_names,
        columns=["PC1_Loading", "PC2_Loading"]
    )

    # --- PLOT 1: Scree & Explained Variance + Top Factor Loadings ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    # Left: Scree / Cumulative Variance
    axes[0].plot(range(1, len(cum_var) + 1), cum_var * 100, marker="o", linewidth=2, color="#1f77b4")
    axes[0].axhline(y=variance_threshold * 100, color="r", linestyle="--", label=f"{variance_threshold*100:.0f}% Cutoff")
    axes[0].axvline(x=n_components, color="black", linestyle=":", label=f"Optimal K={n_components}")
    axes[0].set_title("PCA Scree Plot & Cumulative Variance Explained", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Principal Component Index", fontsize=10)
    axes[0].set_ylabel("Cumulative Variance Explained (%)", fontsize=10)
    axes[0].legend(loc="lower right")

    # Right: Top Loadings for PC1
    top_loadings = loadings["PC1_Loading"].abs().sort_values(ascending=False).head(10)
    top_loadings_signed = loadings.loc[top_loadings.index, "PC1_Loading"]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in top_loadings_signed]
    axes[1].barh(top_loadings_signed.index, top_loadings_signed.values, color=colors)
    axes[1].set_title("Top 10 Biomarker Loadings on PC1", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Eigenvector Weight / Loading", fontsize=10)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("plot_1_pca_variance_and_loadings.png")
    plt.close()
    print("    [Saved]: plot_1_pca_variance_and_loadings.png")

    return X_pca, pca, loadings


def run_clustering_and_phenotyping(X_pca: np.ndarray, k_range=range(2, 8)):
    """
    Compares K-Means and GMM across multiple clusters using BIC, AIC, and Silhouette scores.
    Identifies acute shock anomalies based on low GMM posterior log-likelihood.
    """
    print("\n" + "="*60)
    print("[*] STEP 2: Geometric (K-Means) vs. Probabilistic (GMM) Clustering")
    print("="*60)

    bic_scores, aic_scores = [], []
    silhouette_km, silhouette_gmm = [], []

    for k in k_range:
        # K-Means
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_pca)
        silhouette_km.append(silhouette_score(X_pca, km.labels_))

        # GMM
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(X_pca)
        bic_scores.append(gmm.bic(X_pca))
        aic_scores.append(gmm.aic(X_pca))
        gmm_labels = gmm.predict(X_pca)
        silhouette_gmm.append(silhouette_score(X_pca, gmm_labels))

    # Optimal K selection based on minimum BIC
    best_k = list(k_range)[int(np.argmin(bic_scores))]
    print(f"    Optimal Number of Phenotypes (via BIC minimization): K = {best_k}")

    # Fit final optimal GMM
    best_gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
    best_gmm.fit(X_pca)
    hard_clusters = best_gmm.predict(X_pca)
    responsibilities = best_gmm.predict_proba(X_pca)

    # Anomaly Detection: Points falling in bottom 2% density tail
    log_densities = best_gmm.score_samples(X_pca)
    anomaly_threshold = np.percentile(log_densities, 2.0)
    anomalies = log_densities < anomaly_threshold
    print(f"    Flagged {np.sum(anomalies)} high-risk shock anomalies (Bottom 2% log-density).")

    # --- PLOT 2: Model Selection Metrics ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Left: BIC / AIC curves
    axes[0].plot(list(k_range), bic_scores, marker="o", label="BIC (GMM)", color="#d62728", linewidth=2)
    axes[0].plot(list(k_range), aic_scores, marker="s", label="AIC (GMM)", color="#ff7f0e", linestyle="--")
    axes[0].axvline(x=best_k, color="black", linestyle=":", label=f"Selected K={best_k}")
    axes[0].set_title("Information Criteria vs. Number of Clusters", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[0].set_ylabel("Score (Lower is Better)", fontsize=10)
    axes[0].legend()

    # Right: Silhouette Comparison
    axes[1].plot(list(k_range), silhouette_km, marker="o", label="K-Means", color="#1f77b4", linewidth=2)
    axes[1].plot(list(k_range), silhouette_gmm, marker="^", label="GMM (Hard)", color="#2ca02c", linewidth=2)
    axes[1].set_title("Silhouette Coefficient Comparison", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[1].set_ylabel("Silhouette Score (Higher is Better)", fontsize=10)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plot_2_clustering_metrics.png")
    plt.close()
    print("    [Saved]: plot_2_clustering_metrics.png")

    return hard_clusters, responsibilities, anomalies, best_k, best_gmm


def run_tsne_manifold(X_pca: np.ndarray, clusters: np.ndarray, anomalies: np.ndarray, sample_limit: int = 4000):
    """
    Embeds high-dimensional PCA components into 2D space via t-SNE and plots latent phenotypes.
    """
    print("\n" + "="*60)
    print("[*] STEP 3: Non-Linear Manifold Projection (t-SNE)")
    print("="*60)

    # Subsample if dataset is large for computational speed
    if X_pca.shape[0] > sample_limit:
        idx = np.random.RandomState(42).choice(X_pca.shape[0], sample_limit, replace=False)
    else:
        idx = np.arange(X_pca.shape[0])

    X_sub = X_pca[idx]
    clusters_sub = clusters[idx]
    anomalies_sub = anomalies[idx]

    tsne = TSNE(n_components=2, perplexity=35, max_iter=1000, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sub)
    print(f"    Generated 2D t-SNE embedding for {len(idx):,} patients.")

    # --- PLOT 3: 2D Latent Phenotype Manifold ---
    plt.figure(figsize=(10, 7), dpi=300)
    scatter = sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=[f"Phenotype {c+1}" for c in clusters_sub],
        palette="tab10",
        alpha=0.6,
        s=35,
        legend="full"
    )

    # Overlay shock anomalies
    if np.any(anomalies_sub):
        plt.scatter(
            X_tsne[anomalies_sub, 0],
            X_tsne[anomalies_sub, 1],
            facecolors="none",
            edgecolors="#d62728",
            s=90,
            linewidths=1.5,
            label="Acute Outlier / Shock State"
        )

    plt.title("Latent Clinical ICU Phenotypes & Anomaly Profiling (t-SNE)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("plot_3_tsne_phenotypes_anomalies.png")
    plt.close()
    print("    [Saved]: plot_3_tsne_phenotypes_anomalies.png")

    return X_tsne, idx


def run_regularized_modeling(X_scaled: np.ndarray, target: np.ndarray, clusters: np.ndarray, feature_names: list):
    """
    Fits cluster-stratified Lasso (L1) and Ridge (L2) models to identify sparse biomarkers.
    Plots top active predictive biomarkers per discovered phenotype.
    """
    print("\n" + "="*60)
    print("[*] STEP 4: Cluster-Stratified Regularized Modeling (Lasso & Ridge)")
    print("="*60)

    if target is None:
        print("    Target variable not found. Simulating synthetic clinical severity score for validation...")
        target = 0.45 * X_scaled[:, 0] - 0.65 * X_scaled[:, 1] + 0.3 * (X_scaled[:, 2]**2) + np.random.normal(0, 0.25, len(X_scaled))

    alphas = np.logspace(-3, 3, 50)
    unique_clusters = np.unique(clusters)
    cluster_coefs = {}

    for c in unique_clusters:
        mask = (clusters == c)
        X_c, y_c = X_scaled[mask], target[mask]

        if len(X_c) < 30:
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.25, random_state=42)

        # Lasso (L1)
        lasso = LassoCV(alphas=alphas, cv=5, random_state=42).fit(X_tr, y_tr)
        r2_lasso = lasso.score(X_te, y_te)
        selected_feats = np.sum(lasso.coef_ != 0)

        # Ridge (L2)
        ridge = RidgeCV(alphas=alphas, cv=5).fit(X_tr, y_tr)
        r2_ridge = ridge.score(X_te, y_te)

        cluster_coefs[f"Phenotype {c+1}"] = pd.Series(lasso.coef_, index=feature_names)

        print(f"\n--- Phenotype {c+1} (Cohort Size: {np.sum(mask):,}) ---")
        print(f"    Lasso (L1): R^2 = {r2_lasso:.4f} | Optimal alpha = {lasso.alpha_:.4f} | Active Sparse Features = {selected_feats}/{len(feature_names)}")
        print(f"    Ridge (L2): R^2 = {r2_ridge:.4f} | Optimal alpha = {ridge.alpha_:.4f}")

        # Display Top 3 Active Drivers
        top_idx = np.argsort(np.abs(lasso.coef_))[-3:][::-1]
        for i in top_idx:
            if lasso.coef_[i] != 0:
                print(f"      * {feature_names[i]}: Coef = {lasso.coef_[i]:+.4f}")

    # --- PLOT 4: Lasso Biomarker Coefficients across Phenotypes ---
    coef_df = pd.DataFrame(cluster_coefs)
    # Select top 8 features with largest aggregate coefficient variance
    top_biomarkers = coef_df.abs().max(axis=1).sort_values(ascending=False).head(8).index
    plot_df = coef_df.loc[top_biomarkers]

    plt.figure(figsize=(12, 6), dpi=300)
    plot_df.plot(kind="bar", figsize=(12, 5), width=0.8, ax=plt.gca(), colormap="viridis")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Sparse Lasso Biomarker Effect Sizes Across Clinical Phenotypes", fontsize=12, fontweight="bold")
    plt.xlabel("Key Clinical Biomarkers / Vitals", fontsize=10)
    plt.ylabel("Regularized Beta Coefficient", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Phenotype Group")
    plt.tight_layout()
    plt.savefig("plot_4_cluster_biomarkers.png")
    plt.close()
    print("\n    [Saved]: plot_4_cluster_biomarkers.png")


def main():
    parser = argparse.ArgumentParser(description="Clinical ICU Patient Phenotyping Pipeline")
    parser.add_argument("--csv", type=str, default="Sepsis_data.csv", help="Path to input Sepsis CSV dataset")
    parser.add_argument("--id_col", type=str, default="Patient_ID", help="Patient Identifier column name")
    parser.add_argument("--target_col", type=str, default="ICULOS", help="Target regression column (e.g. ICULOS, SepsisLabel)")
    args = parser.parse_args()

    # 1. Ingestion & Preprocessing
    X_scaled, target, feature_names = load_and_preprocess(args.csv, args.id_col, args.target_col)

    # 2. Dimensionality Reduction & Loading Analysis
    X_pca, pca_model, loadings = run_dimensionality_reduction(X_scaled, feature_names)

    # 3. Model Selection, GMM Clustering & Anomaly Profiling
    clusters, responsibilities, anomalies, best_k, gmm_model = run_clustering_and_phenotyping(X_pca)

    # 4. 2D Manifold Non-linear Projection
    X_tsne, sub_indices = run_tsne_manifold(X_pca, clusters, anomalies)

    # 5. Regularized Downstream Biomarker Discovery
    run_regularized_modeling(X_scaled, target, clusters, feature_names)

    print("\n" + "="*60)
    print("[+] Pipeline execution complete. 4 high-resolution plots generated.")
    print("="*60)


if __name__ == "__main__":
    main()