from pathlib import Path
import time
import duckdb

try:
    from .http_utils import request_with_retry
except ImportError:
    from ingestion.http_utils import request_with_retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FULL_FILE = OUTPUT_DIR / "openfoodfacts_full.parquet"
OUTPUT_FILE = OUTPUT_DIR / "openfoodfacts.parquet"

URL = (
    "https://huggingface.co/datasets/openfoodfacts/"
    "product-database/resolve/main/food.parquet?download=true"
)

TARGET_COLUMNS = [
   "code",
   "product_name",
   "product_name_fr",
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


def build_select_expressions(conn: duckdb.DuckDBPyConnection, source_url: str) -> list[str]:
    """Vérifie le schéma distant et construit des expressions SELECT adaptées aux types."""
    schema_rows = conn.execute(f"DESCRIBE SELECT * FROM '{source_url}' LIMIT 1;").fetchall()
    available_cols = {row[0]: row[1] for row in schema_rows}

    if "countries_tags" not in available_cols:
        raise ValueError("La colonne 'countries_tags' est introuvable dans le dataset distant.")

    expressions = []
    for col in TARGET_COLUMNS:
        if col not in available_cols:
            continue

        col_type = available_cols[col].upper()

        if col == "product_name" and "STRUCT" in col_type:
            expressions.append(
                """
                COALESCE(
                    list_extract(list_filter(product_name, lambda x: x.lang = 'main'), 1)['text'],
                    list_extract(list_filter(product_name, lambda x: x.lang = 'fr'), 1)['text'],
                    list_extract(product_name, 1)['text']
                ) AS product_name
                """.strip()
            )
        elif col_type.endswith("[]") or "LIST" in col_type:
            expressions.append(f"array_to_string({col}, ',') AS {col}")
        else:
            expressions.append(f"CAST({col} AS VARCHAR) AS {col}")

    return expressions


def download_and_filter_file():
    print("🔍 Test rapide de connexion et validation du schéma Open Food Facts...")
    conn = duckdb.connect()

    try:
        # 1. Smoke test immédiat : détecte toute erreur de schéma/SQL avant de télécharger
        select_exprs = build_select_expressions(conn, URL)
        select_clause = ",\n                ".join(select_exprs)

        test_query = f"""
            SELECT {select_clause}
            FROM '{URL}'
            WHERE array_to_string(countries_tags, ',') LIKE '%france%'
            LIMIT 1;
        """
        conn.execute(test_query).fetchone()
        print("✅ Connexion et schéma validés avec succès !")
    except Exception as e:
        print(f"\n❌ ERREUR DÉTECTÉE IMMÉDIATEMENT : {e}")
        conn.close()
        return
    finally:
        conn.close()

    # 2. Téléchargement avec indicateur de pourcentage en direct
    print("\n📥 Téléchargement du dataset Open Food Facts...")
    start_time = time.time()

    response = request_with_retry(URL, stream=True, timeout=600)

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    download_pct = 0.0

    try:
        with open(RAW_FULL_FILE, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        download_pct = (downloaded / total_size) * 100
                        print(
                            f"\r⏳ Chargement des données : {download_pct:.1f}% "
                            f"({downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} Mo)",
                            end="",
                        )
                    else:
                        print(
                            f"\r⏳ Données téléchargées : {downloaded / (1024 * 1024):.1f} Mo",
                            end="",
                        )

        print("\n✅ Téléchargement terminé. Filtrage France et sélection des colonnes via DuckDB...")

        # 3. Filtrage France + projection des colonnes utiles
        conn = duckdb.connect()
        export_query = f"""
            COPY (
                SELECT {select_clause}
                FROM '{RAW_FULL_FILE}'
                WHERE array_to_string(countries_tags, ',') LIKE '%france%'
            ) TO '{OUTPUT_FILE}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
        """
        conn.execute(export_query)

        # Récupération des dimensions finales (lignes, colonnes, taille)
        row_count = conn.execute(f"SELECT COUNT(*) FROM '{OUTPUT_FILE}'").fetchone()[0]
        col_count = len(conn.execute(f"DESCRIBE SELECT * FROM '{OUTPUT_FILE}'").fetchall())
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        elapsed = time.time() - start_time
        final_pct = 100.0

        print("\n" + "=" * 50)
        print(f"📁 Fichier généré        : {OUTPUT_FILE}")
        print(f"📐 Dimensions (L x C)    : {row_count:,} lignes × {col_count} colonnes")
        print(f"💾 Taille sur disque     : {file_size_mb:.2f} Mo (compression ZSTD)")
        print(f"📈 Chargement des données : {final_pct:.1f}%")
        print(f"⏱️  Temps total écoulé    : {elapsed:.1f} s")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Erreur pendant le traitement : {e}")
    finally:
        conn.close()
        if RAW_FULL_FILE.exists():
            RAW_FULL_FILE.unlink()


if __name__ == "__main__":
    download_and_filter_file()