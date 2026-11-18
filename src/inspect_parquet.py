from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def inspect_parquet(path: Path) -> None:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata

    print("\n" + "=" * 80)
    print(path.relative_to(PROJECT_ROOT))
    print("=" * 80)
    print(f"Taille fichier : {format_bytes(path.stat().st_size)}")
    print(f"Lignes         : {metadata.num_rows:,}")
    print(f"Colonnes       : {metadata.num_columns}")
    print(f"Row groups     : {metadata.num_row_groups}")

    print("\nSchéma :")
    for field in parquet_file.schema_arrow:
        print(f"  - {field.name}: {field.type}")

    preview = parquet_file.read_row_group(0)
    preview_names = [field.name for field in parquet_file.schema_arrow][:20]
    preview = preview.select(preview_names).slice(0, 5)
    print("\nAperçu des 5 premières lignes (20 premières colonnes) :")
    print(preview.to_pandas().to_string(index=False))

    print("\nValeurs nulles (50 premières colonnes) :")
    for column_index, column in enumerate(list(parquet_file.schema_arrow)[:50]):
        known_count = True
        null_count = 0

        for row_group in range(metadata.num_row_groups):
            column_metadata = metadata.row_group(row_group).column(column_index)
            statistics = column_metadata.statistics
            if statistics is None or statistics.null_count is None:
                known_count = False
                break
            null_count += statistics.null_count

        null_count_text = f"{null_count:,}" if known_count else "inconnu"
        print(f"  - {column.name}: {null_count_text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspecte les fichiers Parquet présents dans un dossier."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Fichier ou dossier Parquet à examiner.",
    )
    args = parser.parse_args()

    target = args.path.resolve()
    if not target.exists():
        raise SystemExit(f"Chemin introuvable : {target}")

    if target.is_file():
        files = [target] if target.suffix.lower() == ".parquet" else []
    else:
        files = sorted(target.rglob("*.parquet"))

    if not files:
        raise SystemExit(f"Aucun fichier Parquet trouvé dans : {target}")

    print(f"{len(files)} fichier(s) Parquet trouvé(s)")
    for parquet_path in files:
        inspect_parquet(parquet_path)


if __name__ == "__main__":
    main()
