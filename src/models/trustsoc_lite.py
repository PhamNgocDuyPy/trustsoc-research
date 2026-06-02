from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

from ..config import ProjectConfig
from ..utils import get_logger, save_json
from .model_utils import (
    FeatureBundle,
    build_text_graph_features,
    classification_metrics,
    estimate_parameter_count,
    regression_metrics,
    save_model_bundle,
    train_trust_layer,
)


def train_lite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    model_name: str = "trustsoc_lite",
    include_text: bool = True,
    include_cti: bool = True,
    include_mitre: bool = True,
    include_derg: bool = False,
    include_adversarial: bool = True,
    include_trust_calibration: bool = True,
) -> dict[str, Any]:
    logger = get_logger(config, f"train_{model_name}")
    feature_bundle: FeatureBundle = build_text_graph_features(
        train_df,
        val_df,
        test_df,
        config,
        include_text=include_text,
        include_cti=include_cti,
        include_mitre=include_mitre,
        include_derg=include_derg,
        include_adversarial=include_adversarial,
    )
    start = time.perf_counter()
    threat_model = LinearSVC(class_weight="balanced", max_iter=7000)
    severity_model = LinearSVC(class_weight="balanced", max_iter=7000)
    label_model = LinearSVC(class_weight="balanced", max_iter=7000)
    risk_model = Ridge(alpha=1.0)

    threat_model.fit(feature_bundle.train_matrix, train_df["threat_type"])
    severity_model.fit(feature_bundle.train_matrix, train_df["severity"])
    label_model.fit(feature_bundle.train_matrix, train_df["label"])
    risk_model.fit(feature_bundle.train_matrix, train_df["risk_score"])
    train_seconds = time.perf_counter() - start

    infer_start = time.perf_counter()
    train_preds = (
        threat_model.predict(feature_bundle.train_matrix),
        severity_model.predict(feature_bundle.train_matrix),
        label_model.predict(feature_bundle.train_matrix),
        np.clip(risk_model.predict(feature_bundle.train_matrix), 0.0, 100.0),
    )
    val_preds = (
        threat_model.predict(feature_bundle.val_matrix),
        severity_model.predict(feature_bundle.val_matrix),
        label_model.predict(feature_bundle.val_matrix),
        np.clip(risk_model.predict(feature_bundle.val_matrix), 0.0, 100.0),
    )
    test_preds = (
        threat_model.predict(feature_bundle.test_matrix),
        severity_model.predict(feature_bundle.test_matrix),
        label_model.predict(feature_bundle.test_matrix),
        np.clip(risk_model.predict(feature_bundle.test_matrix), 0.0, 100.0),
    )
    inference_seconds = time.perf_counter() - infer_start

    if include_trust_calibration:
        trust_outputs = train_trust_layer(
            threat_model,
            severity_model,
            label_model,
            risk_model,
            train_df,
            val_df,
            test_df,
            feature_bundle.train_matrix,
            feature_bundle.val_matrix,
            feature_bundle.test_matrix,
            train_preds,
            val_preds,
            test_preds,
        )
    else:
        confidence = np.full(len(test_df), predictions := 0.80, dtype=float)
        trust_outputs = type("TrustOutputsShim", (), {})()
        trust_outputs.trust_score = confidence
        trust_outputs.uncertainty_score = 1.0 - confidence
        trust_outputs.reliability_score = test_df["reliability_score"].fillna(0.7).to_numpy(dtype=float)
        trust_outputs.expected_action = np.where(confidence >= 0.75, "conclude", "investigate")
        trust_outputs.threshold = 0.75
        trust_outputs.calibration_metrics = {
            "coverage": 1.0,
            "accept_precision": float(
                (
                    (test_preds[0] == test_df["threat_type"].to_numpy())
                    & (test_preds[1] == test_df["severity"].to_numpy())
                    & (test_preds[2] == test_df["label"].to_numpy())
                ).mean()
            ),
            "refusal_correctness": 0.0,
            "trust_correctness_correlation": 0.0,
            "overconfidence_rate": 0.0,
            "underconfidence_rate": 0.0,
            "adversarial_overconfidence_penalty": 0.0,
            "consistency_alignment": 0.0,
            "trust_alignment_score": 0.0,
            "ece": 0.0,
            "brier": 0.0,
        }

    predictions = pd.DataFrame(
        {
            "case_id": test_df["case_id"],
            "global_id": test_df["global_id"],
            "threat_true": test_df["threat_type"],
            "threat_pred": test_preds[0],
            "severity_true": test_df["severity"],
            "severity_pred": test_preds[1],
            "label_true": test_df["label"],
            "label_pred": test_preds[2],
            "risk_true": test_df["risk_score"],
            "risk_pred": np.round(test_preds[3], 4),
            "trust_score": np.round(trust_outputs.trust_score, 6),
            "uncertainty_score": np.round(trust_outputs.uncertainty_score, 6),
            "reliability_score": np.round(trust_outputs.reliability_score, 6),
            "expected_action_pred": trust_outputs.expected_action,
            "expected_action_true": test_df["expected_action_target"],
            "adversarial_type": test_df["adversarial_type"],
            "event_text": test_df["event_text"],
        }
    )
    predictions["joint_correct"] = (
        (predictions["threat_true"] == predictions["threat_pred"])
        & (predictions["severity_true"] == predictions["severity_pred"])
        & (predictions["label_true"] == predictions["label_pred"])
    )

    threat_metrics = classification_metrics(test_df["threat_type"].to_numpy(), test_preds[0])
    severity_metrics = classification_metrics(test_df["severity"].to_numpy(), test_preds[1])
    label_metrics = classification_metrics(test_df["label"].to_numpy(), test_preds[2])
    risk_metrics = regression_metrics(test_df["risk_score"].to_numpy(), test_preds[3])

    refusal_mask = predictions["expected_action_pred"] == "refuse"
    escalation_mask = predictions["expected_action_pred"] == "escalate"
    calibration = dict(trust_outputs.calibration_metrics)
    calibration["refusal_accuracy"] = float(
        (predictions.loc[refusal_mask, "expected_action_true"] == "refuse").mean() if refusal_mask.any() else 0.0
    )
    calibration["escalation_accuracy"] = float(
        (predictions.loc[escalation_mask, "expected_action_true"] == "escalate").mean() if escalation_mask.any() else 0.0
    )
    corr = np.corrcoef(predictions["trust_score"], predictions["risk_pred"])[0, 1]
    calibration["trust_risk_alignment"] = float(0.0 if np.isnan(corr) else corr)

    metrics = {
        "model_name": model_name,
        "dataset_rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "threat_type": threat_metrics,
        "severity": severity_metrics,
        "label": label_metrics,
        "risk_score": risk_metrics,
        "calibration": calibration,
        "joint_exact_match": float(predictions["joint_correct"].mean()),
        "efficiency": {
            "train_time_seconds": float(train_seconds),
            "total_inference_seconds": float(inference_seconds),
            "average_latency_seconds_per_sample": float(inference_seconds / max(len(test_df), 1)),
            "feature_count": int(feature_bundle.train_matrix.shape[1]),
            "parameter_count_estimate": int(
                estimate_parameter_count(threat_model)
                + estimate_parameter_count(severity_model)
                + estimate_parameter_count(label_model)
                + estimate_parameter_count(risk_model)
            ),
        },
    }

    model_dir = config.models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = config.predictions_dir / f"predictions_{model_name}.csv"
    metrics_path = config.metrics_dir / f"metrics_{model_name}.json"
    bundle_path = model_dir / f"{model_name}.pkl"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8")
    save_json(metrics_path, metrics)
    save_model_bundle(
        bundle_path,
        {
            "feature_bundle": feature_bundle,
            "threat_model": threat_model,
            "severity_model": severity_model,
            "label_model": label_model,
            "risk_model": risk_model,
            "feature_flags": {
                "include_text": include_text,
                "include_cti": include_cti,
                "include_mitre": include_mitre,
                "include_derg": include_derg,
                "include_adversarial": include_adversarial,
                "include_trust_calibration": include_trust_calibration,
            },
            "threshold": trust_outputs.threshold,
            "metrics": metrics,
            "prediction_path": str(prediction_path),
        },
    )
    logger.info("Saved %s metrics to %s", model_name, metrics_path)
    return metrics
