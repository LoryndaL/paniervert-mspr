import os
from pathlib import Path

import pandas as pd
import requests

try:
    import pyarrow  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Merci d'installer 'pyarrow' pour écrire le fichier Parquet: pip install pyarrow"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = OUTPUT_DIR / "rappelconso.parquet"

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/rappelconso-v2-gtin-espaces/records"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_records(limit: int = 100, offset: int = 0):
    params = {"limit": limit, "offset": offset}
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload.get("results", [])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for offset in range(0, 1000, 100):
        batch = fetch_records(limit=100, offset=offset)
        if not batch:
            break
        records.extend(batch)
        print(f"Récupéré: {len(records)} enregistrements")

    if not records:
        print("Aucune donnée récupérée.")
        return

    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Fichier enregistré : {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
