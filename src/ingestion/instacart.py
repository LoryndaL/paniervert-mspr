import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SLUG = "psparks/instacart-market-basket-analysis"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"
PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "instacart_parquet"
LOCAL_KAGGLE_CANDIDATES = [
    PROJECT_ROOT / "kaggle.json",
    PROJECT_ROOT / "src" / "ingestion" / "kaggle.json",
]


def dataset_is_available() -> bool:
    return DATA_DIR.exists() and any(DATA_DIR.rglob("*.csv"))


def ensure_kaggle_credentials(): 
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    username = os.environ.get("KAGGLE_USERNAME")
    api_key = os.environ.get("KAGGLE_KEY")
    if username and api_key:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        token_path = kaggle_dir / "kaggle.json"
        token_path.write_text(json.dumps({"username": username, "key": api_key}), encoding="utf-8")
        return True

    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if token_path.exists():
        return True

    for candidate in LOCAL_KAGGLE_CANDIDATES:
        if candidate.exists():
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            target = kaggle_dir / "kaggle.json"
            target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
            return True

    return False


def download_dataset():
    if dataset_is_available():
        print(f"Dataset Instacart déjà présent dans : {DATA_DIR}")
        return

    if not ensure_kaggle_credentials():
        print(
            "Aucun identifiant Kaggle trouvé et aucun fichier local Instacart détecté.\n"
            "Pour télécharger les données :\n"
            "  export KAGGLE_USERNAME='ton_user'\n"
            "  export KAGGLE_KEY='ta_clé_api'\n"
            "ou : kaggle auth login\n"
            "Ensuite relance le script."
        )
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "Le package Kaggle n'est pas installé.\n"
            "Installe-le avec : pip install kaggle"
        ) from exc

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(DATA_DIR), unzip=True, quiet=False)


def export_to_parquet():
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans {DATA_DIR}. Le téléchargement n'a pas produit de données.")
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        target = PARQUET_DIR / f"{csv_file.stem}.parquet"
        df.to_parquet(target, index=False)
        print(f"Exporté : {target}")


def main():
    download_dataset()
    export_to_parquet()

    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.csv")):
        print(f"Données Kaggle disponibles dans : {DATA_DIR}")
    else:
        print("Aucune donnée Instacart disponible pour l'instant.")

    if PARQUET_DIR.exists() and any(PARQUET_DIR.rglob("*.parquet")):
        print(f"Fichiers Parquet exportés dans : {PARQUET_DIR}")


if __name__ == "__main__":
    main()
