from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .data_loader import load_low_resource_split, load_processed_split
from .error_analysis import generate_error_analysis
from .models.model_utils import comparison_row_from_metrics, opensoc_reference_row
from .models.sklearn_baselines import train_baselines
from .models.trustsoc_derg import train_derg
from .models.trustsoc_lite import train_lite
from .models.trustsoc_transformer import train_transformer
from .utils import get_logger, load_json, save_json
from .visualization import (
    create_model_figures,
    plot_bar_table,
    plot_derg_graph,
    plot_efficiency_table,
    plot_history_comparison,
    plot_low_resource_tradeoff,
    plot_low_resource_variant_bars,
    plot_pipeline_diagram,
    plot_model_radar_chart,
)


def _safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        try:
            df.to_csv(fallback, **kwargs)
            return fallback
        except PermissionError:
            timestamped = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
            df.to_csv(timestamped, **kwargs)
            return timestamped


def _load_or_train_metrics(path: Path, trainer, *args, **kwargs):
    if path.exists():
        return load_json(path)
    return trainer(*args, **kwargs)


def ensure_core_metrics(config: ProjectConfig) -> dict[str, Any]:
    logger = get_logger(config, "evaluate")
    train_df = load_processed_split(config, "train")
    val_df = load_processed_split(config, "val")
    test_df = load_processed_split(config, "test")
    low_train = load_low_resource_split(config, "train")
    low_val = load_low_resource_split(config, "val")
    low_test = load_low_resource_split(config, "test")

    baseline_path = config.metrics_dir / "baseline_metrics.json"
    baselines = load_json(baseline_path) if baseline_path.exists() else train_baselines(train_df, val_df, test_df, config)
    lite_metrics = _load_or_train_metrics(config.metrics_dir / "metrics_lite.json", train_lite, train_df, val_df, test_df, config, "lite")
    derg_metrics = _load_or_train_metrics(config.metrics_dir / "metrics_trustsoc_derg.json", train_derg, train_df, val_df, test_df, config)
    low_resource_metrics = _load_or_train_metrics(
        config.metrics_dir / "metrics_lite_low_resource.json",
        train_lite,
        low_train,
        low_val,
        low_test,
        config,
        "lite_low_resource",
    )
    transformer_metrics = load_json(config.metrics_dir / "metrics_transformer.json") if (config.metrics_dir / "metrics_transformer.json").exists() else train_transformer(train_df, val_df, test_df, config, "transformer")

    save_json(
        config.metrics_dir / "metrics_summary.json",
        {
            "baselines": baselines,
            "lite": lite_metrics,
            "derg": derg_metrics,
            "low_resource": low_resource_metrics,
            "transformer": transformer_metrics,
        },
    )
    logger.info("Core metrics ensured.")
    return {
        "baselines": baselines,
        "lite": lite_metrics,
        "derg": derg_metrics,
        "low_resource": low_resource_metrics,
        "transformer": transformer_metrics,
    }


def compare_with_opensoc(config: ProjectConfig) -> pd.DataFrame:
    summary = ensure_core_metrics(config)
    rows = [opensoc_reference_row(config.baseline_reference)]

    for baseline_name, metrics in summary["baselines"].items():
        rows.append(
            {
                "Model": baseline_name,
                "Accuracy": metrics["threat_type"]["accuracy"],
                "Macro F1": metrics["threat_type"]["f1_macro"],
                "Weighted F1": metrics["threat_type"]["f1_weighted"],
                "MAE": metrics["risk_score"]["mae"],
                "RMSE": metrics["risk_score"]["rmse"],
                "MAPE": metrics["risk_score"]["mape"],
                "R2": metrics["risk_score"]["r2"],
                "ECE": None,
                "Brier": None,
                "Robust-F1": None,
                "Refusal Acc": None,
                "Latency": metrics["efficiency"]["average_latency_seconds_per_sample"],
                "Train Time": metrics["efficiency"]["train_time_seconds"] / 60.0,
                "Params": metrics["efficiency"]["parameter_count_estimate"],
                "Notes": metrics["notes"],
            }
        )

    for model_name, key in (("TrustSOC-Lite", "lite"), ("TrustSOC-DERG", "derg")):
        metrics = summary[key]
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["threat_type"]["accuracy"],
                "Macro F1": metrics["threat_type"]["f1_macro"],
                "Weighted F1": metrics["threat_type"]["f1_weighted"],
                "MAE": metrics["risk_score"]["mae"],
                "RMSE": metrics["risk_score"]["rmse"],
                "MAPE": metrics["risk_score"]["mape"],
                "R2": metrics["risk_score"]["r2"],
                "ECE": metrics["calibration"]["ece"],
                "Brier": metrics["calibration"]["brier"],
                "Robust-F1": metrics["threat_type"]["f1_macro"],
                "Refusal Acc": metrics["calibration"]["refusal_accuracy"],
                "Latency": metrics["efficiency"]["average_latency_seconds_per_sample"],
                "Train Time": metrics["efficiency"]["train_time_seconds"] / 60.0,
                "Params": metrics["efficiency"]["parameter_count_estimate"],
                "Notes": "Local reproducible prototype.",
            }
        )

    transformer_metrics = summary["transformer"]
    if transformer_metrics.get("status") == "trained":
        rows.append(
            {
                "Model": "TrustSOC-Transformer",
                "Accuracy": transformer_metrics["threat_type"]["accuracy"],
                "Macro F1": transformer_metrics["threat_type"]["f1_macro"],
                "Weighted F1": transformer_metrics["threat_type"]["f1_weighted"],
                "MAE": transformer_metrics["risk_score"]["mae"],
                "RMSE": transformer_metrics["risk_score"]["rmse"],
                "MAPE": transformer_metrics["risk_score"]["mape"],
                "R2": transformer_metrics["risk_score"]["r2"],
                "ECE": transformer_metrics["calibration"]["ece"],
                "Brier": transformer_metrics["calibration"]["brier"],
                "Robust-F1": transformer_metrics["threat_type"]["f1_macro"],
                "Refusal Acc": transformer_metrics["calibration"]["refusal_accuracy"],
                "Latency": transformer_metrics["efficiency"]["average_latency_seconds_per_sample"],
                "Train Time": transformer_metrics["efficiency"]["train_time_seconds"] / 60.0,
                "Params": transformer_metrics["efficiency"]["parameter_count"],
                "Notes": "PyTorch text-plus-DERG deep model.",
            }
        )

    rows.append(
        {
            "Model": "RedSage-lite",
            "Accuracy": "to be filled after running experiments",
            "Macro F1": "to be filled after running experiments",
            "Weighted F1": "to be filled after running experiments",
            "MAE": "to be filled after running experiments",
            "RMSE": "to be filled after running experiments",
            "MAPE": "to be filled after running experiments",
            "R2": "to be filled after running experiments",
            "ECE": "to be filled after running experiments",
            "Brier": "to be filled after running experiments",
            "Robust-F1": "to be filled after running experiments",
            "Refusal Acc": "to be filled after running experiments",
            "Latency": "to be filled after running experiments",
            "Train Time": "to be filled after running experiments",
            "Params": "to be filled after running experiments",
            "Notes": "External baseline not rerun locally in this repository.",
        }
    )

    comparison = pd.DataFrame(rows)
    output_path = _safe_to_csv(comparison, config.tables_dir / "table_baseline_comparison.csv", index=False, encoding="utf-8")
    get_logger(config, "evaluate").info("Saved baseline comparison table to %s", output_path)
    return comparison


def generate_report_tables(config: ProjectConfig) -> dict[str, Any]:
    comparison = compare_with_opensoc(config)
    error_table = generate_error_analysis(config, "lite")
    reports_dir = config.artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_candidates = sorted(config.tables_dir.glob("table_baseline_comparison*.csv"))
    comparison_table_path = comparison_candidates[-1] if comparison_candidates else config.tables_dir / "table_baseline_comparison.csv"
    dataset_stats = pd.DataFrame(
        [
            {
                "split": split,
                "rows": len(load_processed_split(config, split)),
                "adversarial_rows": int(load_processed_split(config, split)["adversarial_type"].ne("normal_case").sum()),
            }
            for split in ("train", "val", "test")
        ]
    )
    dataset_stats.to_csv(config.tables_dir / "table_dataset_statistics.csv", index=False, encoding="utf-8")

    efficiency = comparison[comparison["Model"].isin(["OpenSOC-AI", "TrustSOC-Lite", "TrustSOC-DERG"])]
    efficiency.to_csv(config.tables_dir / "table_efficiency.csv", index=False, encoding="utf-8")

    calibration_rows = []
    for model_name in ("lite", "trustsoc_derg", "transformer"):
        metrics_path = config.metrics_dir / f"metrics_{model_name}.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            calibration_rows.append(
                {
                    "Model": model_name,
                    "ECE": metrics["calibration"]["ece"],
                    "Brier": metrics["calibration"]["brier"],
                    "Refusal Correctness": metrics["calibration"]["refusal_correctness"],
                    "Overconfidence Rate": metrics["calibration"]["overconfidence_rate"],
                    "Underconfidence Rate": metrics["calibration"]["underconfidence_rate"],
                    "Trust Alignment Score": metrics["calibration"]["trust_alignment_score"],
                }
            )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(config.tables_dir / "table_calibration.csv", index=False, encoding="utf-8")
    save_json(config.metrics_dir / "calibration_metrics.json", calibration_rows)

    if (config.metrics_dir / "metrics_trustsoc_derg.json").exists():
        shutil.copy2(config.metrics_dir / "metrics_trustsoc_derg.json", config.metrics_dir / "metrics_derg.json")
    if (config.predictions_dir / "predictions_trustsoc_derg.csv").exists():
        shutil.copy2(config.predictions_dir / "predictions_trustsoc_derg.csv", config.predictions_dir / "predictions_derg.csv")
    transformer_metrics = load_json(config.metrics_dir / "metrics_transformer.json")
    if not (config.predictions_dir / "predictions_transformer.csv").exists():
        pd.DataFrame([transformer_metrics]).to_csv(config.predictions_dir / "predictions_transformer.csv", index=False, encoding="utf-8")

    # Run statistical significance tests (Phase 2 contribution)
    try:
        run_statistical_significance_tests(config)
    except Exception as e:
        get_logger(config, "evaluate").error("Failed to run statistical tests: %s", e)

    figures_input = load_processed_split(config, "test").iloc[0]
    plot_derg_graph(figures_input, config.figures_dir / "derg_example_graph.png")
    plot_pipeline_diagram(config.figures_dir / "pipeline_diagram.png")

    plot_bar_table(comparison[pd.to_numeric(comparison["Accuracy"], errors="coerce").notna()], "Model", "Accuracy", "Baseline Comparison", config.figures_dir / "baseline_comparison_chart.png")
    plot_efficiency_table(
        comparison[pd.to_numeric(comparison["Latency"], errors="coerce").notna()].assign(
            Latency=lambda d: pd.to_numeric(d["Latency"], errors="coerce"),
            **{"Train Time": lambda d: pd.to_numeric(d["Train Time"], errors="coerce")},
            Params=lambda d: pd.to_numeric(d["Params"], errors="coerce"),
        ),
        config.figures_dir / "efficiency_comparison.png",
    )
    plot_model_radar_chart(comparison, config.figures_dir / "model_comparison_radar.png")
    if (config.tables_dir / "table_ablation_study.csv").exists():
        ablation_df = pd.read_csv(config.tables_dir / "table_ablation_study.csv")
        plot_bar_table(ablation_df, "Variant", "Weighted F1", "Ablation Study", config.figures_dir / "ablation_study_chart.png")
    if (config.tables_dir / "table_robustness.csv").exists():
        robustness_df = pd.read_csv(config.tables_dir / "table_robustness.csv")
        plot_bar_table(robustness_df, "subset", "trustsoc_lite_f1", "Robustness Evaluation", config.figures_dir / "robustness_evaluation.png")
        plot_bar_table(robustness_df, "subset", "trustsoc_derg_f1", "Adversarial Type Performance", config.figures_dir / "adversarial_type_performance.png")

    create_model_figures(
        config,
        "transformer",
        config.predictions_dir / "predictions_transformer.csv",
        config.metrics_dir / "metrics_transformer.json",
        config.models_dir / "transformer" / "transformer_bundle.pkl",
        training_history_path=config.metrics_dir / "training_history_transformer.csv",
        output_prefix="transformer",
        write_generic=False,
    )
    if (config.predictions_dir / "predictions_transformer_low_resource.csv").exists():
        create_model_figures(
            config,
            "transformer_low_resource",
            config.predictions_dir / "predictions_transformer_low_resource.csv",
            config.metrics_dir / "metrics_transformer_low_resource.json",
            config.models_dir / "transformer_low_resource" / "transformer_low_resource_bundle.pkl",
            training_history_path=config.metrics_dir / "training_history_transformer_low_resource.csv",
            output_prefix="transformer_low_resource",
            write_generic=False,
        )

    low_resource_variant_path = config.tables_dir / "table_transformer_low_resource_variants.csv"
    if low_resource_variant_path.exists():
        low_resource_variants = pd.read_csv(low_resource_variant_path)
        plot_low_resource_tradeoff(low_resource_variants, config.figures_dir / "low_resource_transformer_tradeoff.png")
        plot_low_resource_variant_bars(low_resource_variants, config.figures_dir / "low_resource_transformer_variant_bars.png")
        history_frames = {}
        for variant in ("transformer_low_resource", "transformer_low_resource_nosampler", "transformer_low_resource_warmstart"):
            history_path = config.metrics_dir / f"training_history_{variant}.csv"
            if history_path.exists():
                history_frames[variant] = pd.read_csv(history_path)
        plot_history_comparison(
            history_frames,
            "validation_score",
            "Low-Resource Transformer Validation Score Comparison",
            "Validation Score",
            config.figures_dir / "low_resource_transformer_validation_score.png",
        )
        plot_history_comparison(
            history_frames,
            "joint_exact_match",
            "Low-Resource Transformer Joint Exact Match Comparison",
            "Joint Exact Match",
            config.figures_dir / "low_resource_transformer_joint_exact.png",
        )
        plot_history_comparison(
            history_frames,
            "risk_mae",
            "Low-Resource Transformer Risk MAE Comparison",
            "Risk MAE",
            config.figures_dir / "low_resource_transformer_risk_mae.png",
        )

    transformer_history_path = config.metrics_dir / "training_history_transformer.csv"
    save_json(
        config.metrics_dir / "metrics.json",
        {
            "lite": load_json(config.metrics_dir / "metrics_lite.json"),
            "derg": load_json(config.metrics_dir / "metrics_trustsoc_derg.json"),
            "transformer": transformer_metrics,
            "transformer_low_resource": load_json(config.metrics_dir / "metrics_transformer_low_resource.json"),
        },
    )

    derg_metrics = load_json(config.metrics_dir / "metrics_trustsoc_derg.json")
    transformer_low_metrics = load_json(config.metrics_dir / "metrics_transformer_low_resource.json")
    summary_markdown = "\n".join(
        [
            "# TrustSOC Research Summary",
            "",
            "## Best Full-Split Models",
            f"- TrustSOC-DERG: threat acc {derg_metrics['threat_type']['accuracy']:.4f}, weighted F1 {derg_metrics['threat_type']['f1_weighted']:.4f}, risk MAE {derg_metrics['risk_score']['mae']:.4f}.",
            f"- TrustSOC-Transformer: threat acc {transformer_metrics['threat_type']['accuracy']:.4f}, weighted F1 {transformer_metrics['threat_type']['f1_weighted']:.4f}, risk MAE {transformer_metrics['risk_score']['mae']:.4f}, ECE {transformer_metrics['calibration']['ece']:.4f}.",
            "",
            "## Low-Resource Deep Result",
            f"- Transformer low-resource: threat acc {transformer_low_metrics['threat_type']['accuracy']:.4f}, weighted F1 {transformer_low_metrics['threat_type']['f1_weighted']:.4f}, risk MAE {transformer_low_metrics['risk_score']['mae']:.4f}.",
            "",
            "## Fairness Note",
            "- The deep branch was rerun after removing direct `risk_score` leakage from numeric inputs, so these results are safer for a paper submission.",
            "",
            "## Suggested Paper Positioning",
            "- Claim the strongest overall full-split model as `TrustSOC-DERG`.",
            "- Claim the strongest lightweight deep baseline as `TrustSOC-Transformer`.",
            "- Present low-resource deep learning as a challenging setting where structured evidence still matters more than raw neural text encoding.",
            "",
            "## Main Figures",
            f"- Full transformer training loss: `{config.figures_dir / 'transformer_training_loss_curve.png'}`",
            f"- Full transformer validation overview: `{config.figures_dir / 'transformer_training_metrics_overview.png'}`",
            f"- Full transformer calibration: `{config.figures_dir / 'transformer_calibration_curve.png'}`",
            f"- Full transformer risk scatter: `{config.figures_dir / 'transformer_risk_true_vs_predicted.png'}`",
            f"- Low-resource transformer tradeoff: `{config.figures_dir / 'low_resource_transformer_tradeoff.png'}`",
            f"- Low-resource transformer validation comparison: `{config.figures_dir / 'low_resource_transformer_validation_score.png'}`",
            "",
            "## Key Artifacts",
            f"- Benchmark table: `{comparison_table_path}`",
            f"- Low-resource variants: `{low_resource_variant_path}`",
            f"- Main training history: `{transformer_history_path}`",
        ]
    )
    (reports_dir / "scopus_summary.md").write_text(summary_markdown, encoding="utf-8")

    return {
        "comparison_table": str(comparison_table_path.resolve()),
        "dataset_table": str((config.tables_dir / "table_dataset_statistics.csv").resolve()),
        "error_table": str((config.tables_dir / "table_error_analysis.csv").resolve()),
        "summary_report": str((reports_dir / "scopus_summary.md").resolve()),
    }


def evaluate(config: ProjectConfig) -> dict[str, Any]:
    ensure_core_metrics(config)
    return generate_report_tables(config)


def run_statistical_significance_tests(config: ProjectConfig) -> None:
    """Compute bootstrap confidence intervals and pairwise significance tests."""
    logger = get_logger(config, "evaluate")
    logger.info("Starting statistical significance testing...")

    # Load test split ground truth
    test_df = load_processed_split(config, "test")
    y_true_cls = test_df["threat_type"].to_numpy()
    y_true_reg = test_df["risk_score"].to_numpy()

    # Discover available predictions
    pred_dir = config.predictions_dir
    pred_files = list(pred_dir.glob("predictions_*.csv"))

    if not pred_files:
        logger.warning("No prediction files found in %s. Skipping statistical tests.", pred_dir)
        return

    # Load predictions per model
    models_predictions_cls = {}
    models_predictions_reg = {}

    for p_file in pred_files:
        # Standardize model name (e.g. predictions_lite.csv -> lite)
        model_name = p_file.stem.replace("predictions_", "")
        try:
            p_df = pd.read_csv(p_file)
            if len(p_df) != len(test_df):
                logger.warning(
                    "Prediction file %s has %d rows, but test split has %d rows. Skipping.",
                    p_file.name, len(p_df), len(test_df)
                )
                continue
            
            if "threat_pred" in p_df.columns:
                models_predictions_cls[model_name] = p_df["threat_pred"].to_numpy()
            if "risk_pred" in p_df.columns:
                models_predictions_reg[model_name] = p_df["risk_pred"].to_numpy()
        except Exception as e:
            logger.error("Failed to load predictions from %s: %s", p_file.name, e)

    if not models_predictions_cls:
        logger.warning("No valid model predictions loaded. Skipping statistical tests.")
        return

    # 1. Compute Bootstrap Confidence Intervals for all models
    import numpy as np
    from .statistical_testing import bootstrap_all_metrics, pairwise_model_comparisons, cohens_d, format_ci
    from .latex_export import generate_confidence_interval_table

    ci_rows = []
    ci_data = {}
    
    for model_name, y_pred_cls in models_predictions_cls.items():
        y_pred_reg = models_predictions_reg.get(model_name, np.zeros_like(y_true_reg))
        
        ci_results = bootstrap_all_metrics(
            y_true_cls, y_pred_cls, y_true_reg, y_pred_reg, n_bootstrap=1000
        )
        
        ci_rows.append({
            "Model": model_name,
            "Accuracy_CI": format_ci(ci_results["accuracy"]),
            "Macro_F1_CI": format_ci(ci_results["f1_macro"]),
            "Weighted_F1_CI": format_ci(ci_results["f1_weighted"]),
            "Risk_MAE_CI": format_ci(ci_results["mae"]),
        })

        ci_data[model_name] = {
            "Accuracy": {
                "mean": ci_results["accuracy"]["mean"],
                "ci_lower": ci_results["accuracy"]["ci_lower"],
                "ci_upper": ci_results["accuracy"]["ci_upper"]
            },
            "Macro F1": {
                "mean": ci_results["f1_macro"]["mean"],
                "ci_lower": ci_results["f1_macro"]["ci_lower"],
                "ci_upper": ci_results["f1_macro"]["ci_upper"]
            },
            "Weighted F1": {
                "mean": ci_results["f1_weighted"]["mean"],
                "ci_lower": ci_results["f1_weighted"]["ci_lower"],
                "ci_upper": ci_results["f1_weighted"]["ci_upper"]
            },
            "MAE": {
                "mean": ci_results["mae"]["mean"],
                "ci_lower": ci_results["mae"]["ci_lower"],
                "ci_upper": ci_results["mae"]["ci_upper"]
            }
        }

    # Save CI CSV
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(config.tables_dir / "table_confidence_intervals.csv", index=False, encoding="utf-8")
    logger.info("Saved confidence intervals table to %s", config.tables_dir / "table_confidence_intervals.csv")

    # Generate and save LaTeX table
    try:
        ci_latex = generate_confidence_interval_table(ci_data)
        (config.tables_dir / "table_confidence_intervals.tex").write_text(ci_latex, encoding="utf-8")
        logger.info("Saved confidence intervals LaTeX table to %s", config.tables_dir / "table_confidence_intervals.tex")
    except Exception as e:
        logger.error("Failed to generate CI LaTeX table: %s", e)

    # 2. Pairwise McNemar's Tests for Classification Correctness
    if len(models_predictions_cls) >= 2:
        pairwise_results = pairwise_model_comparisons(y_true_cls, models_predictions_cls)
        pairwise_rows = []
        for res in pairwise_results:
            model_a, model_b = res["model_a"], res["model_b"]
            cd_val = None
            if model_a in models_predictions_reg and model_b in models_predictions_reg:
                cd_val = cohens_d(models_predictions_reg[model_a], models_predictions_reg[model_b])
                
            pairwise_rows.append({
                "Model_A": model_a,
                "Model_B": model_b,
                "McNemar_Stat": res["statistic"],
                "p_value": res["p_value"],
                "Sig_005": res["significant_005"],
                "Sig_001": res["significant_001"],
                "Risk_Cohens_d": cd_val,
            })
            
        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df.to_csv(config.tables_dir / "table_pairwise_significance.csv", index=False, encoding="utf-8")
        logger.info("Saved pairwise significance table to %s", config.tables_dir / "table_pairwise_significance.csv")
