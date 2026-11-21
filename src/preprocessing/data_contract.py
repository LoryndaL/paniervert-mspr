from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class UserFeaturesContract(pa.DataFrameModel):
    user_id: int = pa.Field(ge=0)
    total_orders: float = pa.Field(ge=0, nullable=False)
    avg_basket_size: float = pa.Field(ge=0, nullable=False)
    reorder_rate: float = pa.Field(ge=0, le=1, nullable=True)
    organic_ratio: float = pa.Field(ge=0, le=1, nullable=True)

    class Config:
        coerce = True
        strict = False


def validate_user_features(input_path: Path | None = None) -> dict:
    input_path = input_path or PROCESSED_DIR / "user_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df = pd.read_parquet(input_path)
    optional_columns = [
        "days_since_prior_order_mean",
        "days_since_prior_order",
        "avg_days_between",
    ]
    work = df.copy()
    for column in optional_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    for column in ["total_orders", "avg_basket_size"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    validated = UserFeaturesContract.validate(work, lazy=True)
    report = {
        "input_file": str(input_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(validated)),
        "columns": list(validated.columns),
        "contract": "UserFeaturesContract",
        "status": "passed",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_DIR / "data_contract_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Contrat Pandera validé : %s", report_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print(json.dumps(validate_user_features(), ensure_ascii=False, indent=2))
