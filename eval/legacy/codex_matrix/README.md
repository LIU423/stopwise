# Legacy exploratory Codex policy-classification matrix

Date: 2026-09-10
Cases: 12 handcrafted cases
Runs per configuration: 1

These files are retained for provenance. They are **not an effectiveness benchmark** and are not evidence that StopWise improves decision conversations.

The run tested agreement with small, mostly obvious handcrafted policy labels. Every configuration scored 91.7% action accuracy and disagreed with the same ambiguous `goal-drift` reference label. One run per configuration cannot estimate variance or establish model rankings. It also does not measure decision quality, search cost, redundant turns, or user outcomes.

The stored `summary.json` uses the evaluator terminology that existed at the time. In particular, its `premature_intervention_rate` field came from the retired coarse definition that treated any non-`NO_INTERVENTION` action as premature whenever action-changing information remained. Do not use that field for current claims. The current evaluator distinguishes false intervention, premature `COMMIT`, and unsafe `DEFER`, and allows a correct `FOCUS` while relevant uncertainty remains.

Raw predictions, the original 12 cases, and their fixed-size output schema remain alongside the summary. The active regression suite is [`../../policy/cases.jsonl`](../../policy/cases.jsonl), and the end-to-end design is documented in [`../../controlled/README.md`](../../controlled/README.md).

Do not re-run a broad matrix until the revised cases and evaluation protocol justify the cost.
