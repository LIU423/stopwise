"""Synchronize generated StopWise prompt adapters with the canonical policy."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "prompts" / "custom_instruction.md"
COMPACT = ROOT / "prompts" / "custom_instruction_compact.md"
SKILL = ROOT / "skills" / "stopwise" / "SKILL.md"

SKILL_PREFIX = """---
name: stopwise
description: Use when the user explicitly asks for StopWise or wants help deciding whether more comparison or information search is still action-relevant.
---

# StopWise

This optional Skill is an installation wrapper around the StopWise core strategy. It does not add capabilities beyond that strategy. If an equivalent StopWise system or custom instruction is already active, do not load this wrapper too.

<!-- BEGIN GENERATED CORE POLICY -->
"""
SKILL_SUFFIX = "<!-- END GENERATED CORE POLICY -->\n"


def rendered_assets() -> dict[Path, str]:
    policy = CANONICAL.read_text(encoding="utf-8").strip() + "\n"
    return {
        COMPACT: policy,
        SKILL: SKILL_PREFIX + policy + SKILL_SUFFIX,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when a generated adapter is out of sync",
    )
    args = parser.parse_args()

    expected = rendered_assets()
    stale = [
        path
        for path, content in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if args.check:
        if stale:
            parser.error(
                "out-of-sync policy adapters: "
                + ", ".join(str(path.relative_to(ROOT)) for path in stale)
            )
        return 0

    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
