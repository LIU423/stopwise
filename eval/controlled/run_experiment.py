"""Validate and compare paired baseline and StopWise controlled-run logs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from score import aggregate, load_jsonl, load_tasks, score_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline JSONL log")
    parser.add_argument("--stopwise", type=Path, required=True, help="StopWise JSONL log")
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("tasks.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    baseline = load_jsonl(args.baseline)
    stopwise = load_jsonl(args.stopwise)

    baseline_keys = {(run["task_id"], run.get("replicate", 0)) for run in baseline}
    stopwise_keys = {(run["task_id"], run.get("replicate", 0)) for run in stopwise}
    if baseline_keys != stopwise_keys:
        raise SystemExit("Baseline and StopWise logs must contain identical task/replicate keys")

    models = {run["base_model"] for run in [*baseline, *stopwise]}
    if len(models) != 1:
        raise SystemExit("Both conditions must use the same base_model")
    if any(run["condition"] != "baseline" for run in baseline):
        raise SystemExit("Every baseline row must use condition='baseline'")
    if any(run["condition"] != "stopwise" for run in stopwise):
        raise SystemExit("Every StopWise row must use condition='stopwise'")

    scores = [score_run(tasks[run["task_id"]], run) for run in [*baseline, *stopwise]]
    payload = {
        "status": "experimental_infrastructure; not an effectiveness result",
        "base_model": models.pop(),
        "runs": [asdict(score) for score in scores],
        "summary": aggregate(scores),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
