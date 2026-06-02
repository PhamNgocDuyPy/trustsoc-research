# TrustSOC-Research

## Title

TrustSOC: Trust-Calibrated Multi-Evidence Cyber Reasoning Framework for Low-Resource SOC

## Overview

This repository implements a local, reproducible research prototype for TrustSOC — a framework that extends SOC automation by asking not only whether a model can classify a case correctly, but also **whether it knows when it should conclude, investigate, escalate, or refuse**.

The repository is designed for:

- local preprocessing with evidence extraction and DERG construction
- multi-tier model training (TrustSOC-Lite, TrustSOC-DERG, TrustSOC-Transformer)
- trust calibration with adaptive threshold learning
- DERG-style multi-evidence reasoning
- adversarial robustness benchmarking (7 attack types)
- explainability and evidence attribution
- temporal evidence analysis
- statistical significance testing with bootstrap CIs
- paper-ready tables (LaTeX), figures, and case studies

## Research Question

How can SOC reasoning know when to trust itself and when to refuse a conclusion under low-resource and adversarial evidence conditions?

## Contributions

1. A DERG-based multi-evidence representation for SOC alerts with graph-derived features.
2. A trust calibration mechanism with adaptive threshold learning for deciding when to conclude, investigate, escalate, or refuse.
3. A comprehensive adversarial SOC hallucination benchmark with 7 attack types (noise injection, evidence poisoning, evidence suppression, label manipulation, missing CTI, missing MITRE, adversarial cases).
4. Evidence attribution for explainable trust decisions via SHAP and counterfactual analysis.
5. Temporal evidence decay analysis — evidence freshness-aware trust adjustment.
6. A reproducible benchmark suite with bootstrap confidence intervals, McNemar's test, and paper-ready LaTeX tables.

## Dataset

The code uses local files in `data/raw/` from:

- TrustSOC adversarial dataset (JSONL splits)
- CTI sources (OTX, CVE, malicious domains/IPs)
- Microsoft GUIDE dataset (optional)

## Methodology

The pipeline performs:

1. Raw data copy and schema normalization
2. Evidence extraction with threat pattern matching
3. CTI and MITRE ATT&CK matching
4. DERG construction with graph-derived features
5. Model training for TrustSOC-Lite, TrustSOC-DERG, and TrustSOC-Transformer
6. Trust calibration with adaptive threshold learning
7. Robustness evaluation across 7 adversarial attack types
8. Ablation study (10 variants)
9. Statistical significance testing
10. Figure, table, and case study generation

## Architecture

- `TrustSOC-Lite`: sparse text features (TF-IDF) + evidence features + trust calibration
- `TrustSOC-DERG`: text features + DERG graph features + trust calibration
- `TrustSOC-Transformer`: PyTorch text encoder + DERG numeric features + multi-task heads + trust calibration

## DERG Construction

Each case includes alert, incident, entity, account, device, IP, domain, file, CTI, MITRE, and evidence nodes with reliability and contradiction attributes. Graph features (density, centrality, node/edge counts, etc.) are extracted for model input.

## Trust Calibration

The trust layer predicts:

- trust score (calibrated probability of correct prediction)
- uncertainty score
- reliability score
- expected action: conclude, investigate, escalate, refuse

Enhanced with:
- Adaptive threshold learning (replaces fixed thresholds)
- Temperature scaling for post-hoc calibration
- Trust score decomposition for explainability

## Adversarial SOC Hallucination Benchmark

The benchmark evaluates robustness against 7 attack types:

| Attack Type | Description |
|---|---|
| Noise Injection | Adds benign noise text |
| Evidence Poisoning | Injects fake high-reliability CTI |
| Evidence Suppression | Removes all structured evidence |
| Label Manipulation | Contradicts evidence labels |
| Missing CTI | Zeros CTI features |
| Missing MITRE | Zeros MITRE features |
| Adversarial Cases | Natural adversarial samples |

## Installation

```powershell
cd <project_directory>
python -m pip install -r requirements.txt
```

For the Transformer model, also install PyTorch:
```powershell
python -m pip install torch
```

## How to Run

```powershell
python main.py --mode preprocess
python main.py --mode train_baselines
python main.py --mode train_lite
python main.py --mode train_derg
python main.py --mode train_transformer
python main.py --mode evaluate
python main.py --mode compare_opensoc
python main.py --mode ablation
python main.py --mode robustness
python main.py --mode report
```

Or use individual scripts:

```powershell
python scripts/run_preprocess.py
python scripts/train_lite.py
python scripts/train_derg.py
python scripts/evaluate.py
python scripts/compare_with_opensoc.py
python scripts/run_ablation.py
python scripts/run_robustness.py
python scripts/generate_report_tables.py
```

## Configuration

All paths are auto-detected relative to the project root. To override:

```powershell
# Set custom dataset location
$env:TRUSTSOC_DATASET_ROOT = "C:\path\to\your\dataset"
$env:TRUSTSOC_SOURCE_ROOT = "C:\path\to\trustsoc\data"
python main.py --mode preprocess
```

## Results

Results are generated locally and stored in:

- `artifacts/models/` — trained model bundles
- `artifacts/metrics/` — JSON metrics with bootstrap CIs
- `artifacts/predictions/` — per-sample predictions
- `artifacts/figures/` — publication-quality figures
- `artifacts/tables/` — CSV and LaTeX tables
- `artifacts/reports/` — case studies and summaries

## New Modules

| Module | Purpose |
|---|---|
| `src/statistical_testing.py` | Bootstrap CIs, McNemar's test, Friedman test |
| `src/explainability.py` | SHAP, evidence attribution, counterfactual analysis |
| `src/temporal_analysis.py` | CTI freshness scoring, temporal trust adjustment |
| `src/latex_export.py` | Publication-ready LaTeX tables |
| `src/case_study.py` | Representative case selection and analysis |

## Limitations

- The optional transformer level requires PyTorch and may need GPU for efficient training.
- Expected action labels are heuristic targets derived from evidence rules, not human annotations.
- Temporal analysis uses heuristic half-life values; real CTI timestamps may vary.

## Future Work

- Enable a fully evaluated GNN encoder for true graph-based DERG reasoning
- Add human annotations for trust actions
- Rerun OpenSOC-AI and external baselines under the same evaluation harness
- Integrate real-time CTI feeds for temporal analysis validation

## Citation

Citation placeholder for the future paper.
