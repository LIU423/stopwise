"""Benchmark StopWise classifications across Codex models and effort levels."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from evaluate import score


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "eval/cases.json"
PROMPT_PATH = ROOT / "prompts/stopwise_system.md"
SCHEMA_PATH = ROOT / "eval/codex_batch_schema.json"
DEFAULT_OUTPUT = ROOT / "eval/results/codex_matrix"

MODELS = ("gpt-5.6-sol", "gpt-6-astra", "gpt-5.6-luna", "gpt-5.6-terra")
EFFORTS = {"high": "high", "light": "low"}


def build_prompt(cases: list[dict[str, Any]]) -> str:
    blind_cases = [
        {
            "id": case["id"],
            "description": case["description"],
            "messages": case["messages"],
        }
        for case in cases
    ]
    policy = PROMPT_PATH.read_text(encoding="utf-8")
    return f"""Apply the StopWise policy below to every evaluation case.

This is a classification benchmark. Do not edit files, call tools, or perform external research.
Judge each conversation independently. Return exactly one prediction for every case ID, in the
same order. The outer response must match the supplied JSON schema. Keep each reason to one sentence.

<stopwise_policy>
{policy}
</stopwise_policy>

<evaluation_cases>
{json.dumps(blind_cases, ensure_ascii=False, indent=2)}
</evaluation_cases>
"""


def run_one(
    model: str,
    display_effort: str,
    codex_effort: str,
    prompt: str,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    slug = f"{model}__{display_effort}"
    output_path = output_dir / f"{slug}.json"
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{codex_effort}"',
        "--output-schema",
        str(SCHEMA_PATH),
        "--output-last-message",
        str(output_path),
        "--cd",
        str(ROOT),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "model": model,
            "effort": display_effort,
            "status": "error",
            "error": (completed.stderr or completed.stdout)[-2000:],
        }

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "model": model,
            "effort": display_effort,
            "status": "error",
            "error": f"invalid output: {exc}",
        }

    return {
        "model": model,
        "effort": display_effort,
        "status": "ok",
        "predictions": payload["predictions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Re-score existing output files without calling any model.",
    )
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    prompt = build_prompt(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configurations = [
        (model, display_effort, codex_effort)
        for model in MODELS
        for display_effort, codex_effort in EFFORTS.items()
    ]
    results = []
    if args.score_only:
        for model, display_effort, _ in configurations:
            output_path = args.output_dir / f"{model}__{display_effort}.json"
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "model": model,
                    "effort": display_effort,
                    "status": "ok",
                    "predictions": payload["predictions"],
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_one,
                    model,
                    display_effort,
                    codex_effort,
                    prompt,
                    args.output_dir,
                    args.timeout,
                ): (model, display_effort)
                for model, display_effort, codex_effort in configurations
            }
            for future in as_completed(futures):
                model, effort = futures[future]
                result = future.result()
                results.append(result)
                print(f"{model} / {effort}: {result['status']}", flush=True)

    summary_runs = []
    for result in sorted(results, key=lambda item: (item["model"], item["effort"])):
        summary = {key: result[key] for key in ("model", "effort", "status")}
        if result["status"] == "ok":
            metrics = score(cases, result["predictions"])
            signal_matches = sum(
                set(prediction["signals"]) == set(case["expected_signals"])
                for case, prediction in zip(cases, result["predictions"], strict=True)
            )
            summary["metrics"] = {
                "action_accuracy": metrics.action_accuracy,
                "premature_intervention_rate": metrics.premature_intervention_rate,
                "missed_intervention_rate": metrics.missed_intervention_rate,
                "signal_exact_match": signal_matches / len(cases),
            }
        else:
            summary["error"] = result["error"]
        summary_runs.append(summary)

    summary_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(cases),
        "effort_mapping": EFFORTS,
        "runs": summary_runs,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
