# TrustSOC Research Summary

## Best Full-Split Models
- TrustSOC-DERG: threat acc 0.9990, weighted F1 0.9990, risk MAE 0.5316.
- TrustSOC-Transformer: threat acc 0.9795, weighted F1 0.9791, risk MAE 0.4366, ECE 0.1491.

## Low-Resource Deep Result
- Transformer low-resource: threat acc 0.5600, weighted F1 0.5711, risk MAE 3.4958.

## Fairness Note
- The deep branch was rerun after removing direct `risk_score` leakage from numeric inputs, so these results are safer for a paper submission.

## Suggested Paper Positioning
- Claim the strongest overall full-split model as `TrustSOC-DERG`.
- Claim the strongest lightweight deep baseline as `TrustSOC-Transformer`.
- Present low-resource deep learning as a challenging setting where structured evidence still matters more than raw neural text encoding.

## Main Figures
- Full transformer training loss: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\figures\transformer_training_loss_curve.png`
- Full transformer validation overview: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\figures\transformer_training_metrics_overview.png`
- Full transformer calibration: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\figures\transformer_calibration_curve.png`
- Full transformer risk scatter: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\figures\transformer_risk_true_vs_predicted.png`
- Low-resource transformer tradeoff: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\figures\low_resource_transformer_tradeoff.png`
- Low-resource transformer validation comparison: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\figures\low_resource_transformer_validation_score.png`

## Key Artifacts
- Benchmark table: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\tables\table_baseline_comparison_1780313439.csv`
- Low-resource variants: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\tables\table_transformer_low_resource_variants.csv`
- Main training history: `C:\Users\DUY\Downloads\TrustSOCResearch\TrustSOCResearch\artifacts\metrics\training_history_transformer.csv`