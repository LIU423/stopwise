# Controlled end-to-end evaluation scaffold

This scaffold compares a baseline assistant with the **same base assistant + StopWise** on structured decisions. It is experimental infrastructure, not evidence that StopWise improves conversations.

## Design

Each task has multiple alternatives, hidden queryable attributes, explicit primary criteria, an optimal choice, constraint satisfaction labels, and irrelevant or deferable attributes. A run reveals information incrementally and eventually records a choice.

Hold constant across conditions:

- base model and version;
- decoding parameters and system instructions other than StopWise;
- task order, information environment, and user simulator;
- replicate count and token accounting method.

Randomize condition order where the serving setup permits it. Use multiple replicates; do not treat one run per task as a stable estimate.

## Log format

Store one JSON object per task/replicate:

```json
{
  "task_id": "laptop-engineering",
  "replicate": 0,
  "condition": "baseline",
  "base_model": "example-model-version",
  "query_ids": ["laptop-a-cad", "laptop-a-battery", "laptop-b-cad"],
  "chosen_alternative": "A",
  "prompt_tokens": 900,
  "completion_tokens": 220
}
```

The StopWise log uses the identical task/replicate keys and base model with `"condition": "stopwise"`.

## Run

After collecting both logs:

```bash
python eval/controlled/run_experiment.py \
  --baseline path/to/baseline.jsonl \
  --stopwise path/to/stopwise.jsonl \
  --output path/to/comparison.json
```

Score one mixed-condition log directly with:

```bash
python eval/controlled/score.py path/to/runs.jsonl
```

## Metrics and interpretation

The scorer reports optimal-choice accuracy, constraint satisfaction, utility, regret, information-seeking turns, tokens, redundant queries after sufficient information, premature commit, and Action-Changing Rate (ACR).

`required_query_ids` is an oracle annotation for whether necessary information was acquired; committing without all required queries is scored as premature. `action_changing` is a task-author annotation used in ACR. Both should be reviewed independently when extending the task set.

The desired pattern is reduced redundant search with similar or better decision quality and a low premature-commit rate. Fewer turns alone is not success.

## Still needed for an empirical study

- a model adapter or external harness that records the JSONL schema;
- more tasks and independently reviewed oracle annotations;
- preregistered replicate counts and analysis;
- uncertainty intervals and paired statistical comparisons;
- failure review, including disagreements over what information was truly action-changing.
