"""Compatibility entry point for the policy regression evaluator."""

from pathlib import Path
import runpy


_IMPLEMENTATION = runpy.run_path(
    str(Path(__file__).with_name("policy") / "evaluate.py"),
    run_name="stopwise_policy_evaluator",
)

Metrics = _IMPLEMENTATION["Metrics"]
load_cases = _IMPLEMENTATION["load_cases"]
score = _IMPLEMENTATION["score"]
run = _IMPLEMENTATION["run"]
main = _IMPLEMENTATION["main"]


if __name__ == "__main__":
    main()
