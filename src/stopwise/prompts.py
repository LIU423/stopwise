"""Prompt loading with an installed-package fallback."""

from pathlib import Path
import sysconfig


_PROMPT_NAME = "stopwise_system.md"


def _candidate_paths() -> tuple[Path, ...]:
    source_tree = Path(__file__).resolve().parents[2] / "prompts" / _PROMPT_NAME
    installed = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "stopwise"
        / "prompts"
        / _PROMPT_NAME
    )
    return source_tree, installed


def load_system_prompt(path: str | Path | None = None) -> str:
    """Load the StopWise system prompt.

    A custom path is useful for prompt experiments. By default the source-tree
    prompt is preferred, followed by the copy installed with the package.
    """

    if path is not None:
        return Path(path).read_text(encoding="utf-8").strip()

    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()

    searched = ", ".join(str(path) for path in _candidate_paths())
    raise FileNotFoundError(f"StopWise system prompt not found; searched: {searched}")

