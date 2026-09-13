"""Run the deterministic paired smoke experiment; never calls an external model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .conditions import DeterministicAssistantCallback
from .runner import load_tasks, run_paired_experiment, write_jsonl
from .schemas import ExperimentConfig
from .score import aggregate, score_episode
from .statistics import paired_summary, validate_paired_logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("example_config.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-limit", type=int, help="Optional smoke-only task limit")
    args = parser.parse_args()

    config = ExperimentConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    task_path = Path(config.task_file)
    if not task_path.is_absolute():
        task_path = args.config.parent / task_path
    tasks = load_tasks(task_path)
    if config.task_ids is not None:
        by_id = {task.task_id: task for task in tasks}
        unknown = set(config.task_ids) - set(by_id)
        if unknown:
            raise SystemExit(f"unknown task_ids in config: {sorted(unknown)}")
        tasks = [by_id[task_id] for task_id in config.task_ids]
    if args.task_limit is not None:
        if args.task_limit <= 0:
            raise SystemExit("--task-limit must be positive")
        tasks = tasks[: args.task_limit]

    logs = run_paired_experiment(config, tasks, DeterministicAssistantCallback())
    validate_paired_logs(logs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    write_jsonl(logs, raw_path)

    by_task = {task.task_id: task for task in tasks}
    scores = [score_episode(by_task[log.task_id], log) for log in logs]
    assistant_invocations = sum(score.information_turns for score in scores)
    simulator_decisions = assistant_invocations + len(scores)
    payload = {
        "status": "deterministic smoke validation only; not end-to-end effectiveness evidence",
        "experiment_id": config.experiment_id,
        "raw_log": str(raw_path),
        "request_accounting": {
            "external_requests_executed": 0,
            "assistant_callback_invocations": assistant_invocations,
            "user_simulator_decisions": simulator_decisions,
            "estimated_requests_if_both_callbacks_are_remote": assistant_invocations + simulator_decisions,
        },
        "run_scores": [asdict(item) for item in scores],
        "condition_summary": aggregate(scores),
        "paired_summary": paired_summary(
            scores,
            bootstrap_samples=config.bootstrap_samples,
            confidence_level=config.confidence_level,
            seed=config.base_seed,
        ),
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"raw_log": str(raw_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
