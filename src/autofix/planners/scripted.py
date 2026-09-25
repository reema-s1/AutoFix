"""A deterministic planner that replays a fixed script of actions.

Used to exercise the loop, tools and tracing end to end without a model,
and as a reference for what a planner has to implement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..agent import DONE, Action, Usage
from ..tools import ToolResult, ToolSpec

History = list[tuple[Action, ToolResult]]
ScriptStep = Action | Callable[[History], Action]


class ScriptedPlanner:
    model = "scripted"

    def __init__(self, script: Sequence[ScriptStep]) -> None:
        self.script = list(script)
        self.usage = Usage()
        self.briefing = ""
        self.tools: list[ToolSpec] = []
        self.history: History = []
        self._cursor = 0

    def begin(self, briefing: str, tools: list[ToolSpec]) -> None:
        self.briefing, self.tools = briefing, tools
        self.history, self._cursor = [], 0

    def propose(self) -> Action:
        if self._cursor >= len(self.script):
            return _give_up("script exhausted")
        step = self.script[self._cursor]
        self._cursor += 1
        return step(self.history) if callable(step) else step

    def observe(self, action: Action, result: ToolResult) -> None:
        self.history.append((action, result))

    def conclude(self, reason: str) -> Action:
        remaining = [s for s in self.script[self._cursor:] if not callable(s) and s.tool == DONE]
        return remaining[0] if remaining else _give_up(reason)


def _give_up(reason: str) -> Action:
    return Action(DONE, {"root_cause": "unknown", "summary": reason, "fixed": False, "confidence": 0.0})
