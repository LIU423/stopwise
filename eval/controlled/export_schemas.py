"""Export machine-readable JSON Schemas from the authoritative Pydantic models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import ControlledTask, EpisodeLog, ExperimentConfig


def export_schemas(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "task.schema.json": ControlledTask,
        "episode.schema.json": EpisodeLog,
        "experiment.schema.json": ExperimentConfig,
    }
    paths = []
    for filename, model in models.items():
        path = output_dir / filename
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in export_schemas(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
