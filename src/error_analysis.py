from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


def generate_error_analysis(config: ProjectConfig, model_name: str) -> pd.DataFrame:
    predictions = pd.read_csv(config.predictions_dir / f"predictions_{model_name}.csv")
    errors = predictions[~predictions["joint_correct"]].copy()
    if errors.empty:
        summary = pd.DataFrame(
            [
                {
                    "issue": "no_joint_errors",
                    "count": 0,
                    "note": "No joint errors were observed on this split.",
                }
            ]
        )
    else:
        summary = (
            errors.groupby(["threat_true", "threat_pred", "severity_true", "severity_pred"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    out_path = config.tables_dir / "table_error_analysis.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")
    return summary
