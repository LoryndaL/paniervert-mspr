from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROCESSED_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"

CANDIDATE_FEATURES = [
    "n_orders",
    "total_orders",
    "avg_basket",
    "avg_basket_size",
    "avg_days_between",
    "days_since_prior_order_mean",
    "reorder_rate",
    "organic_ratio",
    "share_produce",
    "share_dairy",
]

SKEWED_FEATURES = {
    "n_orders",
    "total_orders",
    "avg_basket",
    "avg_basket_size",
    "avg_days_between",
    "days_since_prior_order_mean",
}

EXCLUDED_COLS = {
    "user_id",
    "order_id",
    "product_id",
    "segment",
    "segment_nom",
    "cluster",
    "pca_1",
    "pca_2",
}


def resolve_user_features_path() -> Path | None:
    """Localise user_features.parquet dans processed/ ou processed/gold/."""
    candidates = [
        PROCESSED_DIR / "user_features.parquet",
        GOLD_DIR / "user_features.parquet",
    ]
    return next((p for p in candidates if p.exists()), None)


def select_features(df: pd.DataFrame) -> list[str]:
    """Sélectionne les variables numériques pertinentes en excluant tout identifiant."""
    selected = [col for col in CANDIDATE_FEATURES if col in df.columns and col not in EXCLUDED_COLS]
    if selected:
        return selected

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in EXCLUDED_COLS]


def prepare_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """Applique log1p sur les variables asymétriques puis StandardScaler."""
    raw_num = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw_num = raw_num.fillna(raw_num.median(numeric_only=True)).fillna(0.0)

    transformed = raw_num.copy()
    for col in features:
        if col in SKEWED_FEATURES:
            transformed[col] = np.log1p(
                np.clip(transformed[col].to_numpy(dtype=np.float64), a_min=0.0, a_max=None)
            )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(transformed)
    return X_scaled, scaler


def evaluate_k_range(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[pd.DataFrame, int]:
    """Évalue k de 2 à 10 via l'inertie (coude), la silhouette et Davies-Bouldin."""
    max_possible_k = min(k_max, len(X_scaled) - 1)
    if max_possible_k < k_min:
        return pd.DataFrame(), max(1, min(2, len(X_scaled)))

    sample_size = min(20_000, len(X_scaled))
    rows: list[dict] = []
    best_k = k_min
    best_sil = -1.0

    for k in range(k_min, max_possible_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)

        if len(np.unique(labels)) > 1:
            sil = float(silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=42))
            db = float(davies_bouldin_score(X_scaled, labels))
        else:
            sil = -1.0
            db = float("nan")

        rows.append(
            {
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4) if not np.isnan(db) else None,
            }
        )
        print(f"k={k:2d} | inertie={inertia:12.1f} | silhouette={sil:.4f} | davies_bouldin={db:.4f}")

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return pd.DataFrame(rows), best_k


def compare_candidate_k(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    features: list[str],
    candidate_ks: list[int] | None = None,
) -> list[dict]:
    """Compare k=2, 4 et 5 (métriques, répartition en % et moyennes par segment)."""
    if candidate_ks is None:
        candidate_ks = [2, 4, 5]

    sample_size = min(20_000, len(X_scaled))
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    comparison_results: list[dict] = []

    print("\n================ COMPARAISON k = 2 vs 4 vs 5 ================")
    for k in candidate_ks:
        if k < 2 or k >= len(X_scaled):
            continue

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)

        sil = float(silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=42))
        db = float(davies_bouldin_score(X_scaled, labels))
        inertia = float(km.inertia_)

        counts = pd.Series(labels).value_counts().sort_index()
        shares_pct = {int(cid): round(float(cnt / len(labels) * 100), 2) for cid, cnt in counts.items()}
        means_by_seg = (
            df.assign(_seg=labels)
            .groupby("_seg")[numeric_cols]
            .mean(numeric_only=True)
            .round(3)
        )

        print(
            f"\n--- k = {k} | inertie = {inertia:,.1f} | "
            f"silhouette = {sil:.4f} | davies_bouldin = {db:.4f} ---"
        )
        print("Répartition (%) :", shares_pct)
        print(means_by_seg.to_string())

        comparison_results.append(
            {
                "k": int(k),
                "inertia": round(inertia, 2),
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "shares_pct": shares_pct,
                "segment_means": {
                    int(idx): {col: float(val) for col, val in row.items()}
                    for idx, row in means_by_seg.to_dict(orient="index").items()
                },
            }
        )

    return comparison_results


def compute_pca_2d(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calcule une projection ACP 2D et retourne (pca_1, pca_2, explained_variance_ratio)."""
    if len(X_scaled) == 0:
        return np.array([]), np.array([]), []

    if X_scaled.shape[1] >= 2 and len(X_scaled) >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        var_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        return coords[:, 0], coords[:, 1], var_ratio

    pca_1 = X_scaled[:, 0]
    pca_2 = np.zeros(len(X_scaled), dtype=np.float64)
    return pca_1, pca_2, [1.0, 0.0]


def build_business_persona(cluster_id: int, row: pd.Series, global_means: pd.Series) -> dict[str, str]:
    """Associe un nom métier unique, une synthèse et une action commerciale à chaque segment."""
    orders_col = next((c for c in ["total_orders", "n_orders"] if c in global_means.index), None)
    bio_col = next((c for c in ["organic_ratio"] if c in global_means.index), None)
    basket_col = next((c for c in ["avg_basket_size", "avg_basket"] if c in global_means.index), None)
    reorder_col = next((c for c in ["reorder_rate"] if c in global_means.index), None)

    high_orders = bool(orders_col and row[orders_col] >= global_means[orders_col])
    high_bio = bool(bio_col and row[bio_col] >= global_means[bio_col])
    large_basket = bool(basket_col and row[basket_col] >= global_means[basket_col])
    high_reorder = bool(reorder_col and row[reorder_col] >= global_means[reorder_col])

    if high_bio and high_orders and large_basket:
        return {
            "nom": f"Ambassadeurs Bio & Grands Paniers (S{cluster_id})",
            "phrase": "Clients très fidèles commandant souvent de gros paniers fortement orientés bio.",
            "action": "Programme VIP bio, avantages fidélité et accès prioritaire aux nouveautés responsables.",
        }
    if high_bio and high_orders:
        return {
            "nom": f"Engagés Bio Réguliers (S{cluster_id})",
            "phrase": "Clients fréquents privilégiant les produits biologiques au quotidien.",
            "action": "Offres lots bio, abonnements récurrents et recommandations personnalisées.",
        }
    if high_bio and not high_orders:
        return {
            "nom": f"Découvreurs Bio Ponctuels (S{cluster_id})",
            "phrase": "Acheteurs sensibles au bio mais encore peu fréquents sur la plateforme.",
            "action": "Bon de réduction sur la prochaine commande bio et relance de réachat ciblée.",
        }
    if large_basket and (high_orders or high_reorder):
        return {
            "nom": f"Grands Paniers Famille (S{cluster_id})",
            "phrase": "Clients réguliers à paniers volumineux avec une part bio modérée.",
            "action": "Paniers groupés hebdomadaires et substitution progressive vers des alternatives bio.",
        }
    if not high_orders and not large_basket:
        return {
            "nom": f"Petits Paniers Occasionnels (S{cluster_id})",
            "phrase": "Clients à faible fréquence et petits paniers nécessitant de la réactivation.",
            "action": "Campagne de réactivation promotionnelle et mise en avant des essentiels à petit prix.",
        }

    return {
        "nom": f"Clients Standard Équilibrés (S{cluster_id})",
        "phrase": "Profil d'achat intermédiaire proche de la moyenne globale.",
        "action": "Animation commerciale croisée et mise en avant de produits bio du quotidien.",
    }


def compare_dbscan(X_scaled: np.ndarray) -> dict:
    """Compare DBSCAN sur un échantillon plafonné (max 20 000 clients)."""
    if len(X_scaled) == 0:
        return {"sample_size": 0, "clusters_found": 0, "noise_pct": 0.0, "silhouette_non_noise": None}

    sample_n = min(20_000, len(X_scaled))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_scaled), size=sample_n, replace=False)
    X_sample = X_scaled[idx]

    labels = DBSCAN(eps=1.2, min_samples=15).fit_predict(X_sample)
    clusters = [c for c in np.unique(labels) if c != -1]
    noise_pct = float(np.mean(labels == -1) * 100)

    sil_non_noise = None
    mask = labels != -1
    if len(clusters) >= 2 and np.sum(mask) > 1 and len(np.unique(labels[mask])) >= 2:
        sil_non_noise = float(
            silhouette_score(
                X_sample[mask],
                labels[mask],
                sample_size=min(10_000, int(np.sum(mask))),
                random_state=42,
            )
        )

    summary = {
        "sample_size": int(sample_n),
        "clusters_found": int(len(clusters)),
        "noise_pct": round(noise_pct, 2),
        "silhouette_non_noise": round(sil_non_noise, 4) if sil_non_noise is not None else None,
    }
    logging.info("Comparaison DBSCAN (échantillon %s) : %s", sample_n, summary)
    return summary


def save_elbow_silhouette_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde les graphiques du coude (inertie) et du score de silhouette."""
    if metrics_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#1b5e20")
    axes[0].set_title("Méthode du coude (Inertie K-Means)")
    axes[0].set_xlabel("Nombre de clusters (k)")
    axes[0].set_ylabel("Inertie")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="s", color="#2e7d32")
    axes[1].set_title("Score de silhouette (échantillon max 20 000)")
    axes[1].set_xlabel("Nombre de clusters (k)")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _draw_confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 2.0,
    facecolor: str = "none",
    edgecolor: str = "black",
    alpha_fill: float = 0.14,
    linewidth: float = 1.6,
) -> None:
    """Trace une ellipse de confiance autour des points (x, y) d'un cluster."""
    if len(x) < 3:
        return

    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.linalg.det(cov) <= 1e-12:
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 1e-6)
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))

    ellipse_fill = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor="none",
        alpha=alpha_fill,
        zorder=2,
    )
    ellipse_border = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=0.85,
        zorder=3,
    )
    ax.add_patch(ellipse_fill)
    ax.add_patch(ellipse_border)


def save_pca_plot(df: pd.DataFrame, pca_var_ratio: list[float], output_path: Path) -> None:
    """Sauvegarde la projection ACP 2D avec ellipses de confiance par segment."""
    if df.empty or "pca_1" not in df.columns or "pca_2" not in df.columns:
        return

    plot_df = df.sample(n=min(2_000, len(df)), random_state=42)

    palette = ["#009E73", "#E69F00", "#D55E00", "#56B4E9", "#CC79A7", "#0072B2"]
    markers = ["o", "^", "s", "D", "P", "v"]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#E5E5E5", linestyle="-", linewidth=0.9, zorder=0)
    ax.axhline(0, color="#222222", linestyle="--", linewidth=1.2, zorder=1)
    ax.axvline(0, color="#222222", linestyle="--", linewidth=1.2, zorder=1)

    for idx, (seg_id, group) in enumerate(sorted(plot_df.groupby("segment"), key=lambda x: x[0])):
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]
        label_name = group["segment_nom"].iloc[0] if "segment_nom" in group.columns else f"Segment {seg_id}"

        x_vals = group["pca_1"].to_numpy(dtype=np.float64)
        y_vals = group["pca_2"].to_numpy(dtype=np.float64)

        ax.scatter(
            x_vals,
            y_vals,
            c=color,
            marker=marker,
            s=28,
            alpha=0.85,
            edgecolors="none",
            label=label_name,
            zorder=4,
        )
        _draw_confidence_ellipse(
            x_vals,
            y_vals,
            ax=ax,
            n_std=2.0,
            facecolor=color,
            edgecolor=color,
            alpha_fill=0.14,
            linewidth=1.5,
        )

    var_1 = f" ({pca_var_ratio[0] * 100:.1f}%)" if len(pca_var_ratio) >= 1 else ""
    var_2 = f" ({pca_var_ratio[1] * 100:.1f}%)" if len(pca_var_ratio) >= 2 else ""

    ax.set_title("Customer Segments - PCA", loc="left", fontsize=13, pad=12)
    ax.set_xlabel(f"PC1{var_1}", fontsize=11)
    ax.set_ylabel(f"PC2{var_2}", fontsize=11)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(title="Segments", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(forced_k: int | None = 4) -> Path | None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = resolve_user_features_path()
    if input_path is None:
        logging.error("Fichier user_features.parquet introuvable. Lance d'abord src/features/build_features.py.")
        return None

    logging.info("Chargement de %s...", input_path)
    df = pd.read_parquet(input_path)
    if df.empty:
        logging.warning("Le fichier user_features.parquet est vide.")
        return None

    features = select_features(df)
    if not features:
        logging.error("Aucune variable numérique exploitable pour le clustering.")
        return None

    logging.info("Variables retenues pour le clustering (sans ID) : %s", features)
    X_scaled, scaler = prepare_matrix(df, features)

    print("\n=== ÉVALUATION k = 2 À 10 (COUDE + SILHOUETTE + DAVIES-BOULDIN) ===")
    metrics_df, best_k = evaluate_k_range(X_scaled, k_min=2, k_max=10)

    ablation_k = compare_candidate_k(df, X_scaled, features, candidate_ks=[2, 4, 5])

    chosen_k = forced_k if forced_k is not None else best_k
    chosen_k = max(1, min(chosen_k, len(df)))
    logging.info("k optimal silhouette=%s | k retenu pour la segmentation finale=%s", best_k, chosen_k)

    km_final = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    df["segment"] = km_final.fit_predict(X_scaled)

    centers_scaled = pd.DataFrame(km_final.cluster_centers_, columns=features)
    centers_original = pd.DataFrame(
        scaler.inverse_transform(centers_scaled.to_numpy()),
        columns=features,
    )
    for col in features:
        if col in SKEWED_FEATURES:
            centers_original[col] = np.expm1(centers_original[col])

    centers_original.index.name = "segment"
    centers_output = MODELS_DIR / "cluster_centers.csv"
    centers_original.to_csv(centers_output)
    logging.info("Centres des clusters enregistrés : %s", centers_output)

    pca_1, pca_2, pca_var_ratio = compute_pca_2d(X_scaled)
    df["pca_1"] = pca_1
    df["pca_2"] = pca_2

    numeric_cols_for_stats = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    global_means = df[numeric_cols_for_stats].mean(numeric_only=True)
    seg_means = df.groupby("segment")[numeric_cols_for_stats].mean(numeric_only=True)
    seg_counts = df["segment"].value_counts().sort_index()

    personas = {
        int(cid): build_business_persona(int(cid), seg_means.loc[cid], global_means)
        for cid in seg_means.index
    }
    df["segment_nom"] = df["segment"].map(lambda c: personas[int(c)]["nom"])

    profile_rows = []
    for cid in seg_means.index:
        cid_int = int(cid)
        nb = int(seg_counts.loc[cid])
        persona = personas[cid_int]
        row = {
            "segment": cid_int,
            "nom_metier": persona["nom"],
            "nb_clients": nb,
            "part_pct": round(nb / len(df) * 100, 2),
            "action_recommandee": persona["action"],
        }
        for col in numeric_cols_for_stats:
            row[f"moy_{col}"] = round(float(seg_means.loc[cid, col]), 3)
            row[f"ecart_global_{col}"] = round(float(seg_means.loc[cid, col] - global_means[col]), 3)
        profile_rows.append(row)

    print(f"\n=== PROFIL FINAL DES {chosen_k} SEGMENTS VS MOYENNE GLOBALE ===")
    print(pd.DataFrame(profile_rows).to_string(index=False))

    dbscan_summary = compare_dbscan(X_scaled)

    gold_output = GOLD_DIR / "segments.parquet"
    df.to_parquet(gold_output, index=False)
    logging.info("Table Gold enregistrée : %s", gold_output)

    save_elbow_silhouette_plot(metrics_df, MODELS_DIR / "elbow_silhouette.png")
    save_pca_plot(df, pca_var_ratio, MODELS_DIR / "pca_segments.png")

    profiles_payload = {
        "best_silhouette_k": int(best_k),
        "chosen_k": int(chosen_k),
        "total_clients": int(len(df)),
        "features": features,
        "pca_explained_variance_ratio": pca_var_ratio,
        "global_means": {col: round(float(global_means[col]), 4) for col in numeric_cols_for_stats},
        "k_evaluation": metrics_df.to_dict(orient="records"),
        "comparison_k_2_4_5": ablation_k,
        "dbscan_comparison": dbscan_summary,
        "segments": [
            {
                "segment": int(cid),
                "nom_metier": personas[int(cid)]["nom"],
                "description": personas[int(cid)]["phrase"],
                "action_recommandee": personas[int(cid)]["action"],
                "nb_clients": int(seg_counts.loc[cid]),
                "part_pct": float(round(seg_counts.loc[cid] / len(df) * 100, 2)),
                "means": {col: round(float(seg_means.loc[cid, col]), 4) for col in numeric_cols_for_stats},
            }
            for cid in seg_means.index
        ],
    }
    profiles_path = MODELS_DIR / "cluster_profiles.json"
    profiles_path.write_text(json.dumps(profiles_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Profils et métriques enregistrés : %s", profiles_path)

    return gold_output


if __name__ == "__main__":
    main(forced_k=3)