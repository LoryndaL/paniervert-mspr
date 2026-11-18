from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "rappelconso.parquet"

URL = (
    "https://data.economie.gouv.fr/api/explore/v2.1/"
    "catalog/datasets/rappelconso-v2-gtin-espaces/"
    "exports/parquet/?lang=fr&timezone=Europe%2FParis"
)


def download_dataset():

    print("Téléchargement du dataset RappelConso...")

    response = requests.get(
        URL,
        stream=True,
        timeout=600
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get("content-length", 0)
    )

    downloaded = 0

    with open(OUTPUT_FILE, "wb") as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:

                file.write(chunk)

                downloaded += len(chunk)

                if total_size:

                    progress = downloaded / total_size * 100

                    print(
                        f"\rProgression : {progress:.1f}%",
                        end=""
                    )

    print("\n✅ Téléchargement terminé")
    print(f"Fichier enregistré : {OUTPUT_FILE}")


if __name__ == "__main__":
    download_dataset()