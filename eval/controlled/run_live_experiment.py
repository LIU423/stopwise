"""Run an explicitly authorized, budget-guarded OpenAI controlled pilot."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .openai_live import (
    OpenAIAssistantCallback,
    OpenAIResponsesTransport,
    OpenAITextOnlyUserCallback,
    PaidRunBudget,
)
from .runner import load_tasks, run_grouped_experiment, write_jsonl
from .schemas import ControlledTask, EpisodeLog, ExperimentConfig, UsageSource
from .score import aggregate, score_episode
from .simulators import TextOnlyUserSimulator
from .statistics import paired_summary, validate_paired_logs


CONDITIONS = ("baseline", "current_prompt", "minimal_prompt")
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.5"
KNOWN_PRICING = {
    "gpt-5.5-2026-04-23": {
        "input_usd_per_million": 5.0,
        "output_usd_per_million": 30.0,
    }
}


def load_config_and_tasks(config_path: Path) -> tuple[ExperimentConfig, list[ControlledTask]]:
    config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    task_path = Path(config.task_file)
    if not task_path.is_absolute():
        task_path = config_path.parent / task_path
    tasks = load_tasks(task_path)
    if config.task_ids is not None:
        by_id = {task.task_id: task for task in tasks}
        unknown = set(config.task_ids) - set(by_id)
        if unknown:
            raise ValueError(f"unknown task_ids in config: {sorted(unknown)}")
        tasks = [by_id[task_id] for task_id in config.task_ids]
    if not tasks:
        raise ValueError("live pilot requires at least one task")
    return config, tasks


def request_limit(config: ExperimentConfig, task_count: int) -> int:
    # At the cap, each information turn has one user-simulator request and one
    # assistant request. Earlier natural choices use no more than this bound.
    return len(CONDITIONS) * task_count * config.replicates * config.max_information_turns * 2


def make_budget(
    config: ExperimentConfig,
    *,
    task_count: int,
    max_cost_usd: float,
    billing_uplift: float,
) -> PaidRunBudget:
    try:
        pricing = KNOWN_PRICING[config.model.exact_model_version]
    except KeyError as exc:
        raise ValueError(
            "no reviewed price is recorded for exact model "
            f"{config.model.exact_model_version!r}"
        ) from exc
    return PaidRunBudget(
        max_cost_usd=max_cost_usd,
        max_requests=request_limit(config, task_count),
        max_input_tokens_per_request=config.model.context_budget,
        max_output_tokens_per_request=config.model.max_output_tokens,
        input_usd_per_million=pricing["input_usd_per_million"],
        output_usd_per_million=pricing["output_usd_per_million"],
        billing_uplift=billing_uplift,
    )


def build_run_manifest(
    config: ExperimentConfig,
    tasks: list[ControlledTask],
    budget: PaidRunBudget,
) -> dict[str, Any]:
    return {
        "manifest_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "authorized paid pilot configuration",
        "experiment_config": config.model_dump(mode="json"),
        "conditions": list(CONDITIONS),
        "task_ids": [task.task_id for task in tasks],
        "task_count": len(tasks),
        "episode_count": len(CONDITIONS) * len(tasks) * config.replicates,
        "request_budget": budget.snapshot(),
        "pricing_source": PRICING_SOURCE,
        "cost_note": (
            "Conservative estimate prices cached input at the full input rate and applies "
            "the configured billing uplift; it is not an exact provider invoice."
        ),
        "evidence_boundary": "pilot model results are not controlled effectiveness evidence",
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_summary(
    config: ExperimentConfig,
    tasks: list[ControlledTask],
    logs: list[EpisodeLog],
    budget: PaidRunBudget,
    raw_path: Path,
) -> dict[str, Any]:
    validate_paired_logs(logs)
    task_by_id = {task.task_id: task for task in tasks}
    scores = [score_episode(task_by_id[log.task_id], log) for log in logs]
    if any(
        log.assistant_request_count > 0
        and log.token_usage_source != UsageSource.PROVIDER_REPORTED
        for log in logs
    ):
        raise ValueError("assistant token usage is not uniformly provider-reported")
    if any(
        log.user_simulator_token_usage_source != UsageSource.PROVIDER_REPORTED for log in logs
    ):
        raise ValueError("user-simulator token usage is not uniformly provider-reported")
    return {
        "status": "pilot model results; not controlled effectiveness evidence",
        "experiment_id": config.experiment_id,
        "provider": config.model.provider,
        "exact_model_version": config.model.exact_model_version,
        "conditions": list(CONDITIONS),
        "task_ids": [task.task_id for task in tasks],
        "replicates": config.replicates,
        "raw_log": str(raw_path),
        "request_and_cost_accounting": budget.snapshot(),
        "token_accounting": {
            "source": "provider_reported",
            "assistant_prompt_tokens": sum(log.token_usage.prompt_tokens for log in logs),
            "assistant_completion_tokens": sum(log.token_usage.completion_tokens for log in logs),
            "user_simulator_prompt_tokens": sum(
                log.user_simulator_token_usage.prompt_tokens for log in logs
            ),
            "user_simulator_completion_tokens": sum(
                log.user_simulator_token_usage.completion_tokens for log in logs
            ),
        },
        "run_scores": [asdict(item) for item in scores],
        "condition_summary": aggregate(scores),
        "paired_summary": paired_summary(
            scores,
            bootstrap_samples=config.bootstrap_samples,
            confidence_level=config.confidence_level,
            seed=config.base_seed,
        ),
        "interpretation": (
            "Small, authored pilot: report descriptively and do not generalize. "
            "尚未形成 StopWise 端到端有效性证据。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--billing-uplift", type=float, default=1.10)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-paid-run", action="store_true")
    args = parser.parse_args()

    config, tasks = load_config_and_tasks(args.config)
    if config.model.provider != "openai":
        raise SystemExit("live CLI currently supports provider='openai' only")
    budget = make_budget(
        config,
        task_count=len(tasks),
        max_cost_usd=args.max_cost_usd,
        billing_uplift=args.billing_uplift,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "config.json"
    raw_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    _write_json(build_run_manifest(config, tasks, budget), manifest_path)

    if args.dry_run:
        print(json.dumps({"config": str(manifest_path), "preflight": budget.snapshot()}, indent=2))
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env} in the process that launches this command")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("install the live dependency: python -m pip install -e '.[openai]'") from exc

    client = OpenAI(api_key=api_key, max_retries=0, timeout=args.timeout_seconds)
    transport = OpenAIResponsesTransport(client, config.model, budget)
    query_branch_by_id: dict[str, str | None] = {}
    for task in tasks:
        for query_id, query in task.queries.items():
            if query_id in query_branch_by_id and query_branch_by_id[query_id] != query.branch_id:
                raise SystemExit(f"query ID is not globally unique: {query_id}")
            query_branch_by_id[query_id] = query.branch_id
    assistant = OpenAIAssistantCallback(
        transport,
        query_branch_by_id=query_branch_by_id,
    )

    partial_logs: list[EpisodeLog] = []

    def checkpoint(logs: list[EpisodeLog]) -> None:
        partial_logs[:] = logs
        write_jsonl(logs, raw_path)

    try:
        logs = run_grouped_experiment(
            config,
            tasks,
            assistant,
            simulator_factory=lambda task, seed: TextOnlyUserSimulator(
                task,
                seed,
                OpenAITextOnlyUserCallback(transport),
            ),
            on_episode=checkpoint,
        )
        summary = build_summary(config, tasks, logs, budget, raw_path)
        _write_json(summary, summary_path)
    except Exception as exc:
        failure = {
            "status": "paid pilot failed or stopped; partial logs are not paired evidence",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_episodes": len(partial_logs),
            "partial_log": str(raw_path) if partial_logs else None,
            "request_and_cost_accounting": budget.snapshot(),
        }
        _write_json(failure, args.output_dir / "failure.json")
        raise

    print(
        json.dumps(
            {
                "config": str(manifest_path),
                "raw_log": str(raw_path),
                "summary": str(summary_path),
                "budget": budget.snapshot(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
