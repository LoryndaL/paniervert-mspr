from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = RAW_DIR / "collection_metadata.json"

REQUIRED_PATHS = {
    "Instacart": RAW_DIR / "instacart_parquet",
    "Open Food Facts": RAW_DIR / "openfoodfacts.parquet",
    "RappelConso": RAW_DIR / "rappelconso.parquet",
    "Agriculture européenne": RAW_DIR / "agriculture_ec",
}


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def parquet_row_count(path: Path) -> int:
    if path.is_file():
        return pq.ParquetFile(path).metadata.num_rows
    return sum(pq.ParquetFile(p).metadata.num_rows for p in path.rglob("*.parquet"))


def main() -> int:
    missing: list[str] = []

    print("=== Contrôle des données brutes J1 ===")
    for name, path in REQUIRED_PATHS.items():
        exists = path.exists() and (path.is_dir() and any(path.rglob("*")) or path.is_file() and path.stat().st_size > 0)
        status = "OK" if exists else "MANQUANT"
        print(f"{name:24s} | {status:9s} | {path.relative_to(PROJECT_ROOT)}")

        if not exists:
            missing.append(name)
            continue

        size_mb = dir_size(path) / (1024 * 1024)
        print(f"{'':24s} | taille={size_mb:.2f} Mo")

        parquet_files = [path] if path.is_file() and path.suffix == ".parquet" else list(path.rglob("*.parquet"))
        if parquet_files:
            rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)
            print(f"{'':24s} | lignes={rows:,}")

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        print(f"Métadonnées             | OK         | {METADATA_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Collecte                  | {metadata.get('collect_at', 'date inconnue')}")
    else:
        print(f"Métadonnées             | MANQUANTE  | {METADATA_PATH.relative_to(PROJECT_ROOT)}")
        missing.append("Métadonnées")

    if missing:
        print("\nÉléments absents : " + ", ".join(missing), file=sys.stderr)
        return 1

    print("\nContrôle J1 réussi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
