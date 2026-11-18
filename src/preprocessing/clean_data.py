from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def clean_column_name(name: object) -> str:
    name = re.sub(r"[^0-9A-Za-zÀ-ÿ]+", "_", str(name).strip())
    return name.strip("_").lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize names and remove duplicate column names.
    df.columns = [clean_column_name(column) for column in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Convert empty strings, nested objects and arrays to stable hashable text values.
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(
            lambda value: (
                None
                if value is None
                or (isinstance(value, str) and not value.strip())
                else value.strip()
                if isinstance(value, str)
                else json.dumps(
                    value.tolist() if hasattr(value, "tolist") else value,
                    ensure_ascii=False,
                    default=str,
                )
                if isinstance(value, (list, dict, tuple))
                or hasattr(value, "tolist")
                else str(value)
            )
        )

    # Remove empty columns and exact duplicate rows.
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def clean_instacart() -> list[Path]:
    input_dir = RAW_DIR / "instacart_parquet"
    output_dir = PROCESSED_DIR / "instacart"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for input_path in sorted(input_dir.glob("*.parquet")):
        df = pd.read_parquet(input_path)
        cleaned = clean_dataframe(df)
        output_path = output_dir / input_path.name
        cleaned.to_parquet(output_path, index=False)
        outputs.append(output_path)
        print(
            f"{input_path.name}: {len(df):,} -> {len(cleaned):,} lignes, "
            f"{len(cleaned.columns)} colonnes"
        )
    return outputs


def clean_openfoodfacts() -> list[Path]:
    input_path = RAW_DIR / "openfoodfacts.parquet"
    if not input_path.exists():
        return []

    output_dir = PROCESSED_DIR / "openfoodfacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    cleaned = clean_dataframe(df)

    # Keep useful fields for a 2026 data/organic project instead of retaining 5,000 sparse columns.
    preferred = [
        "code",
        "product_name",
        "brands",
        "categories",
        "categories_tags",
        "countries",
        "countries_tags",
        "ingredients_text",
        "ingredients_text_fr",
        "allergens_tags",
        "additives_tags",
        "nutriscore_grade",
        "nova_group",
        "ecoscore_grade",
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

    normalized_columns = {clean_column_name(column): column for column in cleaned.columns}
    available = [
        normalized_columns[clean_column_name(name)]
        for name in preferred
        if clean_column_name(name) in normalized_columns
    ]
    selected = cleaned.loc[:, available].copy() if available else cleaned.copy()

    # Remove rows that do not identify a product.
    identification = [column for column in ("code", "product_name", "brands") if column in selected.columns]
    if identification:
        selected = selected.dropna(subset=identification, how="all")

    selected = selected.reset_index(drop=True)
    output_path = output_dir / "openfoodfacts_clean.parquet"
    selected.to_parquet(output_path, index=False)
    print(
        f"openfoodfacts.parquet: {len(df):,} -> {len(selected):,} lignes, "
        f"5423 -> {len(selected.columns)} colonnes retenues"
    )
    return [output_path]


def clean_rappelconso() -> list[Path]:
    input_path = RAW_DIR / "rappelconso.parquet"
    if not input_path.exists():
        return []

    output_dir = PROCESSED_DIR / "rappelconso"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    cleaned = clean_dataframe(df)

    output_path = output_dir / "rappelconso_clean.parquet"
    cleaned.to_parquet(output_path, index=False)
    print(
        f"rappelconso.parquet: {len(df):,} -> {len(cleaned):,} lignes, "
        f"{len(cleaned.columns)} colonnes"
    )
    return [output_path]


def clean_agriculture_json() -> list[Path]:
    input_path = RAW_DIR / "agriculture_ec" / "organic_logo_data.json"
    if not input_path.exists():
        return []

    output_dir = PROCESSED_DIR / "agriculture_ec"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    summary = pd.DataFrame([{
        "source_url": data.get("source_url"),
        "fetched_at": data.get("fetched_at"),
        "title": data.get("title"),
        "h1": data.get("h1"),
        "paragraph_count": len(data.get("paragraphs", [])),
        "link_count": len(data.get("links", [])),
        "section_count": len(data.get("sections", [])),
    }])
    summary = clean_dataframe(summary)

    summary_path = output_dir / "organic_logo_summary.parquet"
    summary.to_parquet(summary_path, index=False)

    sections = pd.DataFrame(data.get("sections", []))
    sections = clean_dataframe(sections)
    sections_path = output_dir / "organic_logo_sections.parquet"
    sections.to_parquet(sections_path, index=False)

    links = pd.DataFrame(data.get("links", []))
    links = clean_dataframe(links)
    links_path = output_dir / "organic_logo_links.parquet"
    links.to_parquet(links_path, index=False)

    print("organic_logo_data.json: résumé, sections et liens nettoyés")
    return [summary_path, sections_path, links_path]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_instacart()
    clean_openfoodfacts()
    clean_rappelconso()
    clean_agriculture_json()

    print(f"Données nettoyées enregistrées dans : {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
