from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from ingestion.instacart import main as collect_instacart
from ingestion.openfoodfacts import download_and_filter_file
from ingestion.rappelconso import download_dataset
from ingestion.scraping import main as collect_scraping

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = RAW_DIR / "collection_metadata.json"


def collect_source(name: str, collector) -> dict:
    started_at = datetime.now(timezone.utc)
    status = "ok"
    error = None
    try:
        collector()
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        logging.exception("Échec de la collecte %s", name)
    return {
        "source": name,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
    }


def path_size_mb(path: Path) -> float:
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(size / (1024 * 1024), 2)


def parquet_row_count(path: Path) -> int:
    if path.is_file() and path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    return sum(
        pq.ParquetFile(p).metadata.num_rows
        for p in path.rglob("*.parquet")
        if p.is_file()
    )


def write_existing_metadata() -> None:
    inventory = {
        "instacart": RAW_DIR / "instacart_parquet",
        "openfoodfacts": RAW_DIR / "openfoodfacts.parquet",
        "rappelconso": RAW_DIR / "rappelconso.parquet",
        "agriculture_ec": RAW_DIR / "agriculture_ec",
    }
    sources = []
    for name, path in inventory.items():
        exists = path.exists() and (
            path.is_file() or any(path.rglob("*"))
        )
        entry = {
            "source": name,
            "status": "ok" if exists else "missing",
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_mb": path_size_mb(path) if exists else 0.0,
        }
        if exists:
            if path.is_file() and path.suffix.lower() == ".parquet":
                entry["row_count"] = parquet_row_count(path)
            elif any(path.rglob("*.parquet")):
                entry["row_count"] = parquet_row_count(path)
        sources.append(entry)

    now = datetime.now(timezone.utc).isoformat()
    payload = {"collect_at": now, "mode": "metadata_only", "sources": sources}
    METADATA_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Métadonnées de collecte enregistrées : %s", METADATA_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collecte ou inventaire des données brutes")
    parser.add_argument("--metadata-only", action="store_true", help="Inventorie les données existantes sans téléchargement")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        write_existing_metadata()
        return

    results = [
        collect_source("instacart", collect_instacart),
        collect_source("openfoodfacts", download_and_filter_file),
        collect_source("rappelconso", download_dataset),
        collect_source("agriculture_ec", collect_scraping),
    ]
    METADATA_PATH.write_text(
        json.dumps({"collect_at": datetime.now(timezone.utc).isoformat(), "sources": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("Métadonnées de collecte enregistrées : %s", METADATA_PATH)

    failures = [r for r in results if r["status"] == "error"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
