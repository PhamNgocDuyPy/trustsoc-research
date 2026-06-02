from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ProjectConfig
from .data_loader import load_processed_split
from .models.trustsoc_lite import train_lite
from .utils import get_logger, save_json


def run_ablation(config: ProjectConfig) -> list[dict[str, Any]]:
    logger = get_logger(config, "ablation")
    train_df = pd.read_csv(config.low_resource_split_paths["train"], low_memory=False)
    val_df = pd.read_csv(config.low_resource_split_paths["val"], low_memory=False)
    test_df = pd.read_csv(config.low_resource_split_paths["test"], low_memory=False)

    variants = [
        ("full_trustsoc", True, True, True, True, True, True),
        ("without_cti", True, False, True, True, True, True),
        ("without_mitre", True, True, False, True, True, True),
        ("without_derg_graph_features", True, True, True, False, True, True),
        ("without_adversarial_features", True, True, True, True, False, True),
        ("without_trust_calibration", True, True, True, True, True, False),
        ("without_text_embedding", False, True, True, True, True, True),
        ("only_tabular_baseline", False, True, True, True, True, True),
        ("only_text_baseline", True, False, False, False, False, True),
        ("derg_plus_trust_calibration", True, True, True, True, True, True),
    ]

    records: list[dict[str, Any]] = []
    baseline_metrics = train_lite(train_df, val_df, test_df, config, model_name="ablation_reference")
    baseline_f1 = baseline_metrics["threat_type"]["f1_weighted"]
    baseline_mae = baseline_metrics["risk_score"]["mae"]

    for name, include_text, include_cti, include_mitre, include_derg, include_adv, include_trust in variants:
        metrics = train_lite(
            train_df,
            val_df,
            test_df,
            config,
            model_name=f"ablation_{name}",
            include_text=include_text,
            include_cti=include_cti,
            include_mitre=include_mitre,
            include_derg=include_derg,
            include_adversarial=include_adv,
            include_trust_calibration=include_trust,
        )
        record = {
            "Variant": name,
            "Accuracy": metrics["threat_type"]["accuracy"],
            "Weighted F1": metrics["threat_type"]["f1_weighted"],
            "MAE": metrics["risk_score"]["mae"],
            "RMSE": metrics["risk_score"]["rmse"],
            "R2": metrics["risk_score"]["r2"],
            "ECE": metrics["calibration"]["ece"],
            "Robust-F1": metrics["threat_type"]["f1_macro"],
            "Delta_F1": metrics["threat_type"]["f1_weighted"] - baseline_f1,
            "Delta_MAE": metrics["risk_score"]["mae"] - baseline_mae,
            "Interpretation": "Derived from local low-resource ablation run.",
        }
        records.append(record)
        logger.info("Ablation variant %s evaluated", name)

    pd.DataFrame(records).to_csv(config.tables_dir / "table_ablation_study.csv", index=False, encoding="utf-8")
    save_json(config.metrics_dir / "ablation_metrics.json", records)
    return records
