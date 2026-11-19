from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROCESSED_DIR / "gold"


def _resolve_orders_path() -> Path | None:
    """Localise orders.parquet (nettoyé en priorité, sinon brut)."""
    candidates = [
        PROCESSED_DIR / "instacart" / "orders.parquet",
        PROJECT_ROOT / "data" / "raw" / "instacart_parquet" / "orders.parquet",
    ]
    return next((p for p in candidates if p.exists()), None)


def _resolve_order_products_path() -> Path | None:
    """Localise order_products__prior.parquet (nettoyé en priorité, sinon brut)."""
    candidates = [
        PROCESSED_DIR / "instacart" / "order_products__prior.parquet",
        PROJECT_ROOT / "data" / "raw" / "instacart_parquet" / "order_products__prior.parquet",
    ]
    return next((p for p in candidates if p.exists()), None)


def _get_columns(conn: duckdb.DuckDBPyConnection, parquet_path: Path) -> set[str]:
    """Retourne les noms de colonnes d'un fichier Parquet."""
    rows = conn.execute(f"DESCRIBE SELECT * FROM '{parquet_path}' LIMIT 1;").fetchall()
    return {row[0] for row in rows}


def build_user_features(
    conn: duckdb.DuckDBPyConnection,
    orders_path: Path,
    order_products_path: Path,
    products_enriched_path: Path | None,
) -> pd.DataFrame:
    """Agrège les variables comportementales par client via DuckDB."""
    logging.info("Agrégation des features clients (user_features) via DuckDB...")

    order_cols = _get_columns(conn, orders_path)
    op_cols = _get_columns(conn, order_products_path)

    days_expr = (
        "AVG(TRY_CAST(o.days_since_prior_order AS DOUBLE)) AS days_since_prior_order_mean"
        if "days_since_prior_order" in order_cols
        else "NULL::DOUBLE AS days_since_prior_order_mean"
    )
    reorder_expr = (
        "AVG(TRY_CAST(op.reordered AS DOUBLE)) AS reorder_rate"
        if "reordered" in op_cols
        else "0.0::DOUBLE AS reorder_rate"
    )

    if products_enriched_path is not None and products_enriched_path.exists():
        prod_cols = _get_columns(conn, products_enriched_path)
        bio_col = "is_eu_organic_certified" if "is_eu_organic_certified" in prod_cols else None

        if bio_col:
            organic_expr = f"AVG(TRY_CAST(COALESCE(p.{bio_col}, 0) AS DOUBLE)) AS organic_ratio"
        else:
            organic_expr = "0.0::DOUBLE AS organic_ratio"

        query = f"""
            SELECT
                CAST(o.user_id AS BIGINT) AS user_id,
                COUNT(DISTINCT o.order_id) AS total_orders,
                COUNT(op.product_id) * 1.0 / NULLIF(COUNT(DISTINCT o.order_id), 0) AS avg_basket_size,
                {days_expr},
                {reorder_expr},
                {organic_expr}
            FROM '{order_products_path}' op
            INNER JOIN '{orders_path}' o
                ON op.order_id = o.order_id
            LEFT JOIN '{products_enriched_path}' p
                ON op.product_id = p.product_id
            WHERE o.user_id IS NOT NULL
            GROUP BY o.user_id
            ORDER BY o.user_id
        """
    else:
        logging.warning("Catalogue enrichi absent : organic_ratio sera initialisé à 0.0.")
        query = f"""
            SELECT
                CAST(o.user_id AS BIGINT) AS user_id,
                COUNT(DISTINCT o.order_id) AS total_orders,
                COUNT(op.product_id) * 1.0 / NULLIF(COUNT(DISTINCT o.order_id), 0) AS avg_basket_size,
                {days_expr},
                {reorder_expr},
                0.0::DOUBLE AS organic_ratio
            FROM '{order_products_path}' op
            INNER JOIN '{orders_path}' o
                ON op.order_id = o.order_id
            WHERE o.user_id IS NOT NULL
            GROUP BY o.user_id
            ORDER BY o.user_id
        """

    df = conn.execute(query).df()
    for col in ["total_orders", "avg_basket_size", "days_since_prior_order_mean", "reorder_rate", "organic_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "organic_ratio" in df.columns:
        df["organic_ratio"] = df["organic_ratio"].fillna(0.0).clip(0.0, 1.0)
    if "reorder_rate" in df.columns:
        df["reorder_rate"] = df["reorder_rate"].fillna(0.0).clip(0.0, 1.0)

    return df


def main() -> Path | None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    orders_path = _resolve_orders_path()
    order_products_path = _resolve_order_products_path()
    products_enriched_path = PROCESSED_DIR / "products_enriched.parquet"

    if orders_path is None or order_products_path is None:
        logging.error("Fichiers Instacart (orders / order_products__prior) introuvables.")
        return None

    conn = duckdb.connect()
    try:
        user_features = build_user_features(
            conn=conn,
            orders_path=orders_path,
            order_products_path=order_products_path,
            products_enriched_path=products_enriched_path if products_enriched_path.exists() else None,
        )
    finally:
        conn.close()

    output_path = PROCESSED_DIR / "user_features.parquet"
    user_features.to_parquet(output_path, index=False)

    gold_copy = GOLD_DIR / "user_features.parquet"
    user_features.to_parquet(gold_copy, index=False)

    logging.info(
        "user_features.parquet généré : %s clients, %s colonnes -> %s",
        f"{len(user_features):,}",
        len(user_features.columns),
        output_path,
    )
    return output_path


if __name__ == "__main__":
    main()