"""The plan -> act -> observe loop.

The loop is deliberately independent of any particular LLM. A
:class:`Planner` proposes one action at a time and is shown the real result
of each action before proposing the next, so every decision after the first
is conditioned on what the agent has just learned. Swapping the planner
(Claude, a scripted planner for tests, a single-shot baseline) changes the
decision-maker but not the harness, which keeps comparisons fair.

The harness, not the model, has the final word on success: after the loop
ends, the full build and test suite are run again and the outcome is
recorded next to what the model *claimed*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import ToolResult, Toolbox, ToolSpec
from .tracing import Tracer

DONE = "declare_done"

DECLARE_DONE_SPEC = ToolSpec(
    DONE,
    "Stop and report. Call this once the build and tests pass (verify with run_tests first), "
    "or when you cannot make further progress. Be honest: set fixed=false if tests still fail.",
    {
        "type": "object",
        "properties": {
            "root_cause": {
                "type": "string",
                "description": "The underlying defect, naming the file and function, e.g. "
                "'ring_push in src/ring.c checks count > cap instead of count >= cap'.",
            },
            "summary": {"type": "string", "description": "What you changed and how you verified it."},
            "fixed": {"type": "boolean", "description": "True only if run_tests last reported all tests passing."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["root_cause", "summary", "fixed", "confidence"],
        "additionalProperties": False,
    },
)

REASONING_PROPERTIES = {
    "belief": {
        "type": "string",
        "description": "Your current hypothesis about the failure and why this action tests or fixes it.",
    },
    "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "How confident you are in that hypothesis (0-1).",
    },
}


def with_reasoning_fields(spec: ToolSpec) -> ToolSpec:
    """Require the model to state its belief and confidence alongside every action.

    This turns the plan into structured data that is traced per step, instead of
    free text that may or may not be emitted.
    """
    schema = dict(spec.input_schema)
    schema["properties"] = {**REASONING_PROPERTIES, **schema.get("properties", {})}
    schema["required"] = ["belief", "confidence", *schema.get("required", [])]
    return ToolSpec(spec.name, spec.description, schema)


def planner_tool_specs(toolbox: Toolbox) -> list[ToolSpec]:
    return [with_reasoning_fields(s) for s in toolbox.specs] + [DECLARE_DONE_SPEC]


@dataclass
class Action:
    tool: str
    args: dict[str, Any]
    belief: str = ""
    confidence: float | None = None
    call_id: str | None = None

    @classmethod
    def from_tool_input(cls, name: str, raw: dict[str, Any], call_id: str | None = None) -> "Action":
        args = dict(raw)
        belief = args.pop("belief", "") if name != DONE else raw.get("root_cause", "")
        confidence = args.get("confidence") if name == DONE else args.pop("confidence", None)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = None
        return cls(name, args, str(belief or ""), confidence, call_id)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.model_calls += other.model_calls


class PlannerError(RuntimeError):
    """The planner could not produce an action (API failure, refusal, ...)."""


class Planner(Protocol):
    usage: Usage

    def begin(self, briefing: str, tools: list[ToolSpec]) -> None:
        """Start a new episode with the initial failure report."""

    def propose(self) -> Action:
        """Return the next action, conditioned on everything observed so far."""

    def observe(self, action: Action, result: ToolResult) -> None:
        """Feed back the result of executing ``action``."""

    def conclude(self, reason: str) -> Action:
        """The step budget is spent: return a final ``declare_done`` without further tool use."""


@dataclass
class StepRecord:
    iter: int
    action: Action
    result: ToolResult


@dataclass
class AgentOutcome:
    run_id: str
    stop_reason: str  # already_passing | declared_done | budget_exhausted | planner_error
    iterations: int
    claimed_fixed: bool | None
    root_cause: str
    summary: str
    verified_fixed: bool
    changes: str
    modified_files: list[str]
    usage: Usage
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def hallucinated_fix(self) -> bool:
        """Claimed success that the harness could not reproduce."""
        return bool(self.claimed_fixed) and not self.verified_fixed


@dataclass
class AgentConfig:
    max_iters: int = 8


def build_briefing(initial: ToolResult, max_iters: int) -> str:
    return (
        "The project's build or test suite is failing. Here is the result of running the "
        "build and full test suite just now:\n\n"
        f"<initial_run>\n{initial.content}\n</initial_run>\n\n"
        f"You have a budget of {max_iters} tool call(s) before you must stop. "
        "Find the root cause, fix it in the code under test, verify with run_tests, "
        f"then call {DONE}."
    )


def run_agent(
    toolbox: Toolbox,
    planner: Planner,
    tracer: Tracer,
    config: AgentConfig | None = None,
) -> AgentOutcome:
    config = config or AgentConfig()
    tracer.emit(
        "run_start",
        project=toolbox.config.root.name,
        max_iters=config.max_iters,
        model=getattr(planner, "model", type(planner).__name__),
    )

    initial = toolbox.call("run_tests", {})
    tracer.emit("observation", iter=0, summary=_first_line(initial.content), data=initial.data)
    if initial.data.get("all_passed"):
        tracer.emit("terminate", iter=0, reason="already_passing", summary="nothing to fix")
        return _finish(toolbox, planner, tracer, "already_passing", 0, None, "", "", [])

    planner.begin(build_briefing(initial, config.max_iters), planner_tool_specs(toolbox))

    steps: list[StepRecord] = []
    final: Action | None = None
    stop_reason = "budget_exhausted"
    for iteration in range(config.max_iters):
        try:
            action = planner.propose()
        except PlannerError as exc:
            tracer.emit("error", iter=iteration, message=str(exc))
            stop_reason = "planner_error"
            break
        _trace_plan(tracer, iteration, action)

        if action.tool == DONE:
            final, stop_reason = action, "declared_done"
            break

        result = toolbox.call(action.tool, action.args)
        tracer.emit(
            "tool_result",
            iter=iteration,
            tool=action.tool,
            is_error=result.is_error,
            summary=_first_line(result.content),
            data=result.data,
        )
        steps.append(StepRecord(iteration, action, result))
        planner.observe(action, result)
    else:
        try:
            final = planner.conclude("step budget exhausted")
            _trace_plan(tracer, len(steps), final)
        except PlannerError as exc:
            tracer.emit("error", iter=len(steps), message=str(exc))

    claimed = final.args.get("fixed") if final else None
    root_cause = final.args.get("root_cause", "") if final else ""
    summary = final.args.get("summary", "") if final else ""
    tracer.emit("terminate", iter=len(steps), reason=stop_reason, summary=summary, root_cause=root_cause)
    return _finish(
        toolbox, planner, tracer, stop_reason, len(steps),
        claimed if isinstance(claimed, bool) else None, root_cause, summary, steps,
    )


def _finish(
    toolbox: Toolbox,
    planner: Planner,
    tracer: Tracer,
    stop_reason: str,
    iterations: int,
    claimed: bool | None,
    root_cause: str,
    summary: str,
    steps: list[StepRecord],
) -> AgentOutcome:
    report = toolbox.test_report()
    verified = bool(report and report.all_passed)
    usage = getattr(planner, "usage", Usage())
    outcome = AgentOutcome(
        run_id=tracer.run_id,
        stop_reason=stop_reason,
        iterations=iterations,
        claimed_fixed=claimed,
        root_cause=root_cause,
        summary=summary,
        verified_fixed=verified,
        changes=toolbox.changes(),
        modified_files=toolbox.modified_files,
        usage=usage,
        steps=steps,
    )
    tracer.emit(
        "run_end",
        stop_reason=stop_reason,
        iterations=iterations,
        claimed_fixed=claimed,
        verified_fixed=verified,
        modified_files=outcome.modified_files,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        model_calls=usage.model_calls,
    )
    return outcome


def _trace_plan(tracer: Tracer, iteration: int, action: Action) -> None:
    tracer.emit(
        "plan",
        iter=iteration,
        belief=action.belief,
        confidence=action.confidence,
        tool=action.tool,
        args=action.args,
    )


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""
