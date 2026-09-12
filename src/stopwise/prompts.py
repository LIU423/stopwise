"""Load StopWise prompt assets from a checkout or installed package."""

from pathlib import Path
import sysconfig


_PROMPT_NAMES = {
    "analyzer": "analyzer_system.md",
    "custom_instruction": "custom_instruction.md",
    "custom_instruction_compact": "custom_instruction_compact.md",
}


def _candidate_paths(prompt_name: str) -> tuple[Path, ...]:
    source_tree = Path(__file__).resolve().parents[2] / "prompts" / prompt_name
    installed = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "stopwise"
        / "prompts"
        / prompt_name
    )
    return source_tree, installed


def load_prompt(kind: str = "analyzer", path: str | Path | None = None) -> str:
    """Load one of the canonical StopWise prompts.

    ``kind`` is ignored when ``path`` is supplied. Source-tree assets are
    preferred, followed by copies installed as package data.
    """

    if path is not None:
        return Path(path).read_text(encoding="utf-8").strip()

    try:
        prompt_name = _PROMPT_NAMES[kind]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROMPT_NAMES))
        raise ValueError(f"unknown prompt kind {kind!r}; choose from: {choices}") from exc

    for candidate in _candidate_paths(prompt_name):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()

    searched = ", ".join(str(candidate) for candidate in _candidate_paths(prompt_name))
    raise FileNotFoundError(f"StopWise prompt not found; searched: {searched}")


def load_system_prompt(path: str | Path | None = None) -> str:
    """Load the structured analyzer prompt (backwards-compatible alias)."""

    return load_prompt("analyzer", path)
