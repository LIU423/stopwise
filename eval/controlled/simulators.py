"""Reproducible user simulators for controlled decision episodes."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from .conditions import AssistantResponse
from .environment import OracleState
from .schemas import ControlledTask, QueryRelevance, StopWiseAction


@dataclass(frozen=True)
class UserAction:
    kind: str
    query_id: str | None = None
    choice: str | None = None


class UserSimulator(Protocol):
    def observe(self, response: AssistantResponse) -> None: ...

    def next_action(self, state: OracleState) -> UserAction: ...


class DeterministicUserSimulator:
    """Scripted, seeded simulator for tests and dry runs.

    It is intentionally simple and policy-responsive. Its results validate state
    transitions and paired logging only; they are not human-behavior evidence.
    """

    def __init__(self, task: ControlledTask, seed: int) -> None:
        self.task = task
        self.seed = seed
        self._rng = Random(seed)
        self._cursor = 0
        self._commit_requested = False
        self._focus_primary = False
        self.deferred_branches: set[str] = set()
        self.skipped_queries: list[str] = []

    def observe(self, response: AssistantResponse) -> None:
        if not response.visible_nudge:
            return
        if response.action == StopWiseAction.COMMIT:
            self._commit_requested = True
        elif response.action == StopWiseAction.FOCUS:
            self._focus_primary = True
        elif response.action == StopWiseAction.DEFER and response.target_branch_id:
            self.deferred_branches.add(response.target_branch_id)

    def next_action(self, state: OracleState) -> UserAction:
        if self._commit_requested:
            return UserAction(kind="choose", choice=self._choose(state))

        while self._cursor < len(self.task.simulator_query_plan):
            query_id = self.task.simulator_query_plan[self._cursor]
            self._cursor += 1
            query = self.task.queries[query_id]
            if query.branch_id in self.deferred_branches:
                self.skipped_queries.append(query_id)
                continue
            if self._focus_primary and query.relevance not in {
                QueryRelevance.PRIMARY,
                QueryRelevance.ROBUSTNESS,
            }:
                self.skipped_queries.append(query_id)
                continue
            if query.relevance in {QueryRelevance.PRIMARY, QueryRelevance.ROBUSTNESS}:
                self._focus_primary = False
            return UserAction(kind="query", query_id=query_id)
        return UserAction(kind="choose", choice=self._choose(state))

    def _choose(self, state: OracleState) -> str:
        candidates = list(state.recommended or state.possible_feasible or self.task.alternatives)
        # The seed only breaks genuine ties, preserving paired reproducibility.
        return candidates[self._rng.randrange(len(candidates))]


class TextOnlyUserModelCallback(Protocol):
    """Future LLM simulator hook.

    Implementations receive the visible transcript and task description only.
    Condition names, StopWise labels, hidden values, and oracle state must not be
    included by the caller.
    """

    def __call__(
        self,
        *,
        task_id: str,
        scenario: str,
        transcript: list[dict[str, str]],
        seed: int,
    ) -> UserAction: ...
