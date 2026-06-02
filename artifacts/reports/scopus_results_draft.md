# Results and Discussion Draft

## Experimental Summary

We evaluated three main families in the local TrustSOCResearch pipeline: a sparse evidence-aware baseline (`TrustSOC-Lite`), a graph-enhanced structured reasoning model (`TrustSOC-DERG`), and a lightweight deep neural text-plus-evidence model (`TrustSOC-Transformer`). All reported deep-learning results below were rerun after removing direct `risk_score` leakage from the model inputs.

## Full-Split Results

The strongest overall model on the full split is `TrustSOC-DERG`, which achieved threat classification accuracy of `0.99896`, weighted F1 of `0.99895`, and risk-score MAE of `0.53157`. This confirms that explicit structured evidence modeling remains highly effective for SOC reasoning when rich graph features are available.

The strongest deep baseline is `TrustSOC-Transformer`, which achieved threat accuracy of `0.97953`, weighted F1 of `0.97913`, severity accuracy of `0.99028`, label accuracy of `0.99098`, risk-score MAE of `0.43664`, and `R2 = 0.99110`. Relative to the user-provided OpenSOC-AI reference in this repository, the transformer is slightly below the reference threat accuracy (`0.97953` vs `0.98000`), but exceeds the reference weighted F1 (`0.97913` vs `0.97200`) and substantially improves regression quality (`MAE 0.43664` vs `1.48`, `RMSE 2.20631` vs `10.465`, `R2 0.99110` vs `0.574`). The deep model is also materially faster, with `0.000742 s/sample` latency and `0.63 min` training time versus the reference `8.79 s/sample` and `55.22 min`.

These results suggest that the deep branch is competitive with OpenSOC-AI on end-task classification and clearly stronger on efficiency and risk estimation, but it does not surpass the graph-based TrustSOC-DERG model on full-split classification accuracy.

## Calibration and Trust

Among the main models, `TrustSOC-Transformer` currently delivers the best calibration quality in the repository, with `ECE = 0.14907`, compared with `0.18088` for `TrustSOC-Lite` and `0.17999` for `TrustSOC-DERG`. Its trust alignment score is `0.73971`, which is higher than both `TrustSOC-Lite` (`0.72610`) and `TrustSOC-DERG` (`0.69419`). This is a meaningful result for the paper narrative because it supports the claim that lightweight neural fusion can improve confidence quality even when it does not win absolute classification performance.

## Low-Resource Results

Low-resource deep learning remains difficult in the current setting. The default low-resource transformer reached threat accuracy `0.56`, weighted F1 `0.57110`, risk-score MAE `3.49576`, and joint exact match `0.52`. Additional variants reveal a tradeoff between classification stability and risk prediction:

- `Transformer-LR-Default`: balanced overall behavior with strong label detection and acceptable risk modeling.
- `Transformer-LR-NoSampler`: slightly higher threat accuracy (`0.58`) but similar weighted F1 and somewhat unstable calibration.
- `Transformer-LR-WarmStart`: the best weighted F1 among the low-resource transformer variants (`0.58664`), but much weaker label accuracy (`0.78`), making it less attractive as the default paper result.

The low-resource findings indicate that structured evidence remains more robust than purely neural encoding in scarce-data conditions. This supports positioning `TrustSOC-DERG` as the primary contribution and `TrustSOC-Transformer` as a strong lightweight deep baseline rather than the final flagship model.

## Recommended Paper Claims

The safest claims supported by the current artifacts are:

- `TrustSOC-DERG` is the strongest overall model in the current repository on the full split.
- `TrustSOC-Transformer` is a competitive lightweight deep baseline that improves calibration and risk estimation while remaining highly efficient.
- Structured evidence graphs remain essential in low-resource SOC reasoning.

The following stronger claims are not yet fully supported and should be avoided without extra experiments:

- "TrustSOC-Transformer fully outperforms OpenSOC-AI on all metrics."
- "The low-resource deep model beats OpenSOC-AI."
- "Human-AI trust alignment is validated with human annotation."

## Remaining Validity Threats

Two limitations should be stated explicitly in the paper draft:

- OpenSOC-AI has not yet been rerun inside the same local harness; the repository currently compares against the user-provided OpenSOC-AI reference values.
- `expected_action_target` is still heuristic rather than human-annotated, so the trust-action analysis should be framed as a proxy evaluation rather than a final human-trust benchmark.
