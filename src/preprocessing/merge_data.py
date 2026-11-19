from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OFF_MAX_REF_PRODUCTS = 50_000
BATCH_SIZE = 5_000
SIMILARITY_THRESHOLD = 0.45

DESIRED_OFF_COLS = [
        "code",
        "product_name",
        "brands",
        "categories",
        "categories_tags",
        "labels_tags",
        "countries_tags",
        "ingredients_text",
        "ingredients_text_fr",
        "allergens_tags",
        "additives_tags",
        "nutriscore_grade",
        "ecoscore_grade",
        "nova_group",
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "sodium_100g",
        "quantity",
        "packaging",
]


def _get_existing_columns(conn: duckdb.DuckDBPyConnection, parquet_path: Path) -> set[str]:
    """Retourne l'ensemble des colonnes présentes dans un fichier Parquet."""
    rows = conn.execute(f"DESCRIBE SELECT * FROM '{parquet_path}' LIMIT 1;").fetchall()
    return {row[0] for row in rows}


def _load_off_reference(conn: duckdb.DuckDBPyConnection, off_path: Path) -> pd.DataFrame:
    """Charge un référentiel Open Food Facts nettoyé et plafonné pour le rapprochement."""
    if not off_path.exists():
        logging.warning("Fichier Open Food Facts nettoyé absent : %s", off_path)
        return pd.DataFrame(columns=["product_name"])

    available = _get_existing_columns(conn, off_path)
    selected = [col for col in DESIRED_OFF_COLS if col in available]
    if "product_name" not in selected:
        logging.warning("Colonne 'product_name' absente de %s", off_path)
        return pd.DataFrame(columns=selected)

    cols_sql = ", ".join(f'"{c}"' for c in selected)
    query = f"""
        SELECT {cols_sql}
        FROM '{off_path}'
        WHERE product_name IS NOT NULL AND TRIM(CAST(product_name AS VARCHAR)) != ''
        LIMIT {OFF_MAX_REF_PRODUCTS * 3}
    """
    ref_df = conn.execute(query).df()
    ref_df["clean_name"] = ref_df["product_name"].fillna("").astype(str).str.lower().str.strip()
    ref_df = ref_df[ref_df["clean_name"] != ""].drop_duplicates(subset=["clean_name"]).reset_index(drop=True)

    if len(ref_df) > OFF_MAX_REF_PRODUCTS:
        ref_df = ref_df.sample(n=OFF_MAX_REF_PRODUCTS, random_state=42).reset_index(drop=True)

    return ref_df


def _load_rappel_brands(conn: duckdb.DuckDBPyConnection, rappel_path: Path) -> list[str]:
    """Extrait les marques uniques nettoyées depuis RappelConso."""
    if not rappel_path.exists():
        logging.warning("Fichier RappelConso nettoyé absent : %s", rappel_path)
        return []

    available = _get_existing_columns(conn, rappel_path)
    brand_col = next(
        (c for c in ["nom_de_la_marque_du_produit", "marque_produit", "marques"] if c in available),
        None,
    )
    if not brand_col:
        logging.warning("Aucune colonne marque trouvée dans %s", rappel_path)
        return []

    brands_df = conn.execute(
        f"""
        SELECT DISTINCT LOWER(TRIM(CAST("{brand_col}" AS VARCHAR))) AS brand
        FROM '{rappel_path}'
        WHERE "{brand_col}" IS NOT NULL
        """
    ).df()

    invalid = {"", "sans marque", "nd", "none", "nan", "null"}
    return [b for b in brands_df["brand"].dropna().tolist() if b not in invalid and len(b) >= 4]


def merge_catalog(products: pd.DataFrame, off_ref: pd.DataFrame, rappel_brands: list[str]) -> pd.DataFrame:
    """Enrichit le catalogue produits Instacart avec Open Food Facts (TF-IDF par lots) et RappelConso."""
    logging.info("Rapprochement du catalogue avec Open Food Facts et RappelConso...")

    out = products.copy()
    name_col = "product_name" if "product_name" in out.columns else out.columns[1]
    out["clean_name"] = out[name_col].fillna("").astype(str).str.lower().str.strip()

    best_idx = np.zeros(len(out), dtype=np.int64)
    best_scores = np.zeros(len(out), dtype=np.float32)

    if len(off_ref) > 0 and len(out) > 0:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, dtype=np.float32)
        tfidf_off = vec.fit_transform(off_ref["clean_name"])

        for start in range(0, len(out), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(out))
            tfidf_batch = vec.transform(out["clean_name"].iloc[start:end])
            sim_batch = cosine_similarity(tfidf_batch, tfidf_off)
            best_idx[start:end] = sim_batch.argmax(axis=1)
            best_scores[start:end] = sim_batch.max(axis=1)

    out["off_match_score"] = best_scores

    if len(off_ref) > 0:
        matched = off_ref.iloc[best_idx].reset_index(drop=True)
        mask_valid = out["off_match_score"].to_numpy() >= SIMILARITY_THRESHOLD
        for col in ["nutriscore_grade", "ecoscore_grade", "nova_group", "allergens_tags", "labels_tags", "categories_tags"]:
            if col in matched.columns:
                out[col] = np.where(mask_valid, matched[col].to_numpy(), None)

    labels_series = (
        out["labels_tags"].fillna("").astype(str).str.lower()
        if "labels_tags" in out.columns
        else pd.Series("", index=out.index)
    )
    name_bio = out["clean_name"].str.contains(
        r"\b(?:organic|bio|ab)\b|agriculture biologique|eu-bio|ecocert",
        regex=True,
        na=False,
    )
    label_bio = labels_series.str.contains(
        r"en:organic|fr:ab-agriculture-biologique|eu-organic|eu:organic",
        regex=True,
        na=False,
    )
    out["is_eu_organic_certified"] = np.where(name_bio | label_bio, 1, 0).astype(np.int8)

    if rappel_brands:
        top_brands = pd.Series(rappel_brands).value_counts().head(500).index
        brand_pattern = "|".join(
            pd.Series(top_brands).str.replace(r"([.*+?^${}()|\[\]\\])", r"\\\1", regex=True)
        )
        out["has_recall_alert"] = out["clean_name"].str.contains(brand_pattern, regex=True, na=False)
    else:
        out["has_recall_alert"] = False

    logging.info(
        "Fusion terminée : %s produits enrichis | %s marqués bio",
        f"{len(out):,}",
        f"{int(out['is_eu_organic_certified'].sum()):,}",
    )
    return out


def main() -> Path | None:
    products_path = PROCESSED_DIR / "instacart" / "products.parquet"
    if not products_path.exists():
        logging.error("Fichier produits nettoyé introuvable : %s. Lance d'abord clean_data.py.", products_path)
        return None

    off_path = PROCESSED_DIR / "openfoodfacts" / "openfoodfacts_clean.parquet"
    rappel_path = PROCESSED_DIR / "rappelconso" / "rappelconso_clean.parquet"
    output_path = PROCESSED_DIR / "products_enriched.parquet"

    conn = duckdb.connect()
    try:
        products = pd.read_parquet(products_path)
        off_ref = _load_off_reference(conn, off_path)
        rappel_brands = _load_rappel_brands(conn, rappel_path)
    finally:
        conn.close()

    merged = merge_catalog(products, off_ref, rappel_brands)
    merged.to_parquet(output_path, index=False)
    logging.info("Catalogue fusionné enregistré : %s", output_path)
    return output_path


if __name__ == "__main__":
    main()