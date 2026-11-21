from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROCESSED_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> dict:
    input_path = GOLD_DIR / "segments.parquet"
    predictions_path = GOLD_DIR / "classification_predictions.parquet"
    if not input_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Fichiers segments ou prédictions introuvables")

    data = pd.read_parquet(input_path)
    predictions = pd.read_parquet(predictions_path)

    subgroup_candidates = [c for c in ["gender", "sex", "region", "age_group"] if c in data.columns]
    report = {
        "status": "not_applicable" if not subgroup_candidates else "evaluated",
        "subgroups_available": subgroup_candidates,
        "groups": {},
    }

    if subgroup_candidates:
        subgroup = subgroup_candidates[0]
        y_true = data["segment"] if "segment" in data.columns else None
        y_pred = predictions["predicted_class"] if "predicted_class" in predictions.columns else None
        if y_true is not None and y_pred is not None and len(y_true) == len(y_pred):
            for group, mask in data[subgroup].groupby(data[subgroup]).groups.items():
                indices = list(mask)
                if len(indices) < 2:
                    continue
                report["groups"][str(group)] = {
                    "count": len(indices),
                    "recall_macro": round(float(recall_score(y_true.iloc[indices], y_pred.iloc[indices], average="macro", zero_division=0)), 4),
                }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_DIR / "fairness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
