"""Paired summaries, bootstrap intervals, binary comparisons, and effect sizes."""

from __future__ import annotations

from math import comb
from random import Random
from statistics import mean, median, stdev
from typing import Any, Iterable

from .schemas import EpisodeLog
from .score import RunScore


PairKey = tuple[str, str, int, int]


def _key(item: EpisodeLog | RunScore) -> PairKey:
    return (item.experiment_id, item.task_id, item.replicate, item.seed)


def validate_paired_logs(logs: Iterable[EpisodeLog]) -> dict[PairKey, dict[str, EpisodeLog]]:
    pairs: dict[PairKey, dict[str, EpisodeLog]] = {}
    for log in logs:
        if log.condition not in {"baseline", "stopwise"}:
            continue
        pair = pairs.setdefault(_key(log), {})
        if log.condition in pair:
            raise ValueError(f"duplicate {log.condition} row for pair {_key(log)}")
        pair[log.condition] = log
    if not pairs:
        raise ValueError("no baseline/stopwise pairs found")
    missing = {key: sorted({"baseline", "stopwise"} - set(pair)) for key, pair in pairs.items() if len(pair) != 2}
    if missing:
        raise ValueError(f"paired conditions missing: {missing}")

    for key, pair in pairs.items():
        baseline, stopwise = pair["baseline"], pair["stopwise"]
        equal_fields = (
            "task_version", "task_definition_hash", "domain", "base_model", "exact_model_version", "assistant_model_config",
            "base_prompt_hash", "assistant_callback_id", "user_simulator_id",
            "user_simulator_version", "max_information_turns",
        )
        for field in equal_fields:
            if getattr(baseline, field) != getattr(stopwise, field):
                raise ValueError(f"pair {key} differs on controlled field {field}")
    return pairs


def pair_scores(scores: Iterable[RunScore]) -> dict[PairKey, dict[str, RunScore]]:
    pairs: dict[PairKey, dict[str, RunScore]] = {}
    for score in scores:
        pair = pairs.setdefault(_key(score), {})
        if score.condition in pair:
            raise ValueError(f"duplicate {score.condition} score for pair {_key(score)}")
        pair[score.condition] = score
    if not pairs:
        raise ValueError("no paired scores found")
    missing = {key: sorted({"baseline", "stopwise"} - set(pair)) for key, pair in pairs.items() if set(pair) != {"baseline", "stopwise"}}
    if missing:
        raise ValueError(f"paired conditions missing: {missing}")
    return pairs


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float], *, samples: int, confidence_level: float, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty paired difference")
    rng = Random(seed)
    boot = [mean(rng.choice(values) for _ in values) for _ in range(samples)]
    alpha = 1 - confidence_level
    return _percentile(boot, alpha / 2), _percentile(boot, 1 - alpha / 2)


def exact_mcnemar_p(discordant_01: int, discordant_10: int) -> float:
    n = discordant_01 + discordant_10
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(discordant_01, discordant_10) + 1)) / (2**n)
    return min(1.0, 2 * tail)


NUMERIC_METRICS = (
    "utility", "regret", "information_turns", "new_information_query_count", "total_conversation_turns", "prompt_tokens",
    "completion_tokens", "total_tokens", "redundant_query_count", "redundant_query_rate",
    "action_changing_rate", "queries_before_sufficiency", "queries_after_sufficiency",
)
BINARY_METRICS = ("optimal_choice", "satisfactory_choice", "constraint_satisfied")


def paired_summary(
    scores: Iterable[RunScore], *, bootstrap_samples: int = 2000, confidence_level: float = 0.95,
    seed: int = 20260912,
) -> dict[str, Any]:
    pairs = pair_scores(scores)
    ordered = [pairs[key] for key in sorted(pairs)]
    result: dict[str, Any] = {
        "pair_count": len(ordered),
        "difference_direction": "stopwise_minus_baseline",
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_samples,
        "inference_note": (
            "Descriptive paired estimates and intervals only; the sample is too small for strong claims."
            if len(ordered) < 20
            else "Paired estimates are reported with bootstrap intervals; interpretation should follow the preregistered analysis."
        ),
        "numeric": {},
        "binary": {},
    }
    for metric_index, metric in enumerate(NUMERIC_METRICS):
        differences = [
            float(getattr(pair["stopwise"], metric)) - float(getattr(pair["baseline"], metric))
            for pair in ordered
        ]
        low, high = bootstrap_mean_ci(
            differences,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + metric_index,
        )
        sd = stdev(differences) if len(differences) > 1 else 0.0
        result["numeric"][metric] = {
            "mean_paired_difference": mean(differences),
            "median_paired_difference": median(differences),
            "paired_differences": differences,
            "bootstrap_ci": [low, high],
            "standardized_effect_size_dz": mean(differences) / sd if sd > 0 else None,
        }
    for metric in BINARY_METRICS:
        baseline = [bool(getattr(pair["baseline"], metric)) for pair in ordered]
        stopwise = [bool(getattr(pair["stopwise"], metric)) for pair in ordered]
        discordant_01 = sum((not left) and right for left, right in zip(baseline, stopwise, strict=True))
        discordant_10 = sum(left and (not right) for left, right in zip(baseline, stopwise, strict=True))
        result["binary"][metric] = {
            "baseline_rate": mean(baseline),
            "stopwise_rate": mean(stopwise),
            "paired_risk_difference": mean(stopwise) - mean(baseline),
            "discordant_baseline0_stopwise1": discordant_01,
            "discordant_baseline1_stopwise0": discordant_10,
            "exact_mcnemar_p": exact_mcnemar_p(discordant_01, discordant_10),
        }

    domains = sorted({pair["baseline"].domain for pair in ordered})
    result["by_domain"] = {
        domain: paired_summary(
            [score for pair in ordered if pair["baseline"].domain == domain for score in pair.values()],
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + index + 1000,
        )
        for index, domain in enumerate(domains)
    } if len(domains) > 1 else {}
    return result
