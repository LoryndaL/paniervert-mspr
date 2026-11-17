import json
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
OUTPUT_PATH = OUTPUT_DIR / "openfoodfacts.parquet"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(
                lambda x: (
                    json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (list, dict))
                    else (None if pd.isna(x) else str(x))
                )
            )
    return df


def main():
    url = "https://world.openfoodfacts.org/api/v2/search"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    response = None
    for page_size in (20, 10, 5, 1):
        params = {"page_size": page_size}
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if response.ok:
            break
        print(f"Tentative page_size={page_size} -> HTTP {response.status_code}")

    if response is None or not response.ok:
        raise SystemExit(f"API indisponible: HTTP {response.status_code if response else 'N/A'}")

    data = response.json()
    products = data.get("products", [])

    if not products:
        print("Aucune donnée trouvée pour cette requête.")
        return

    df = pd.json_normalize(products)
    df = normalize_dataframe(df)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Fichier enregistré : {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
