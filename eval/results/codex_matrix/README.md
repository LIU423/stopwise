# Codex model matrix — single-run baseline

Date: 2026-09-10  
Cases: 12  
Runs per configuration: 1  
Effort mapping: `high` -> `high`, `light` -> Codex `low`

The benchmark passed blind case payloads to each model: `expected_actions`, `expected_signals`, and `action_changing_information_remains` were removed before inference. Each run classified all 12 cases in one batch.

| Model | Effort | Action accuracy | Premature intervention | Missed intervention | Signal exact match |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-luna | high | 91.7% | 0.0% | 0.0% | 91.7% |
| gpt-5.6-luna | light | 91.7% | 0.0% | 0.0% | 100.0% |
| gpt-5.6-sol | high | 91.7% | 0.0% | 0.0% | 91.7% |
| gpt-5.6-sol | light | 91.7% | 0.0% | 0.0% | 91.7% |
| gpt-5.6-terra | high | 91.7% | 0.0% | 0.0% | 91.7% |
| gpt-5.6-terra | light | 91.7% | 0.0% | 0.0% | 91.7% |
| gpt-6-astra | high | 91.7% | 0.0% | 0.0% | 83.3% |
| gpt-6-astra | light | 91.7% | 0.0% | 0.0% | 83.3% |

## Main disagreement

Every configuration disagreed with the `FOCUS` reference action on `goal-drift`. Seven selected `COMMIT`; gpt-5.6-luna/light selected `DEFER`. Their reasons consistently argued that the irrelevant research could not change a safe, refundable hotel choice whose primary criteria were already resolved.

This result exposes a policy-boundary question: should goal drift always redirect with `FOCUS`, or should a stable underlying decision permit `COMMIT`? The current handcrafted label retains `FOCUS` to match the original evaluation specification.

## Interpretation

- No configuration intervened on a case marked as containing action-changing information.
- No configuration stayed silent on a case requiring intervention.
- `high` did not improve action accuracy in this small single-run batch.
- Signal exact match is deliberately strict: an additional plausible signal counts as a mismatch.
- One run per configuration is not enough to estimate variance or establish model rankings.

Raw predictions and machine-readable aggregate metrics are stored alongside this report. Re-score them without new model calls using:

```bash
conda run -n stopwise python eval/run_codex_matrix.py --score-only
```

