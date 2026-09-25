from autofix.agent import DONE, Action, AgentConfig, PlannerError, Usage, planner_tool_specs, run_agent
from autofix.planners import ScriptedPlanner
from autofix.redaction import Redactor
from autofix.tools import Toolbox
from autofix.tracing import MemorySink, Tracer

FIX = "@@ -4,1 +4,1 @@\n-    return a - b;\n+    return a + b;\n"


def make(toy_project):
    sink = MemorySink()
    tracer = Tracer(run_id="test", sinks=[sink], redactor=Redactor(env={}))
    return Toolbox(toy_project), tracer, sink


def done(fixed: bool, cause: str = "add() subtracts") -> Action:
    return Action(DONE, {"root_cause": cause, "summary": "s", "fixed": fixed, "confidence": 0.9}, cause, 0.9)


def test_loop_investigates_fixes_and_verifies(toy_project):
    toolbox, tracer, sink = make(toy_project)
    planner = ScriptedPlanner(
        [
            Action("read_file", {"path": "src/calc.c"}, "add is wrong", 0.4),
            Action("apply_patch", {"path": "src/calc.c", "diff": FIX}, "uses - instead of +", 0.8),
            Action("run_tests", {}, "verify", 0.9),
            done(True),
        ]
    )
    outcome = run_agent(toolbox, planner, tracer, AgentConfig(max_iters=8))

    assert outcome.stop_reason == "declared_done"
    assert outcome.iterations == 3
    assert outcome.verified_fixed and outcome.claimed_fixed
    assert not outcome.hallucinated_fix
    assert outcome.modified_files == ["src/calc.c"]
    assert "+    return a + b;" in outcome.changes
    assert outcome.root_cause == "add() subtracts"
    assert sink.steps() == [
        "run_start", "observation",
        "plan", "tool_result", "plan", "tool_result", "plan", "tool_result",
        "plan", "terminate", "run_end",
    ]
    plan = sink.events[2]
    assert plan["belief"] == "add is wrong" and plan["confidence"] == 0.4 and plan["tool"] == "read_file"


def test_each_decision_sees_previous_result(toy_project):
    """The planner's second decision is conditioned on the first observation."""
    toolbox, tracer, _ = make(toy_project)

    def decide(history):
        _, result = history[-1]
        line = next(l for l in result.content.splitlines() if "return" in l)
        assert "a - b" in line
        return Action("apply_patch", {"path": "src/calc.c", "diff": FIX}, "saw a - b", 0.9)

    planner = ScriptedPlanner([Action("grep_source", {"pattern": "return"}), decide, done(True)])
    assert run_agent(toolbox, planner, tracer).verified_fixed


def test_budget_exhaustion_asks_planner_to_conclude(toy_project):
    toolbox, tracer, sink = make(toy_project)
    planner = ScriptedPlanner([Action("run_build", {}), Action("run_build", {}), done(False, "not sure")])
    outcome = run_agent(toolbox, planner, tracer, AgentConfig(max_iters=2))
    assert outcome.stop_reason == "budget_exhausted"
    assert outcome.iterations == 2
    assert outcome.claimed_fixed is False and outcome.root_cause == "not sure"
    assert not outcome.verified_fixed
    assert sink.events[-2]["reason"] == "budget_exhausted"


def test_hallucinated_fix_is_detected(toy_project):
    toolbox, tracer, _ = make(toy_project)
    outcome = run_agent(toolbox, ScriptedPlanner([done(True)]), tracer)
    assert outcome.claimed_fixed and not outcome.verified_fixed
    assert outcome.hallucinated_fix


def test_already_passing_project_short_circuits(toy_project):
    calc = toy_project.root / "src" / "calc.c"
    calc.write_text(calc.read_text().replace("a - b", "a + b"))
    toolbox, tracer, _ = make(toy_project)
    planner = ScriptedPlanner([])
    outcome = run_agent(toolbox, planner, tracer)
    assert outcome.stop_reason == "already_passing" and outcome.verified_fixed
    assert planner.briefing == ""  # never consulted


def test_tool_errors_are_fed_back_not_fatal(toy_project):
    toolbox, tracer, _ = make(toy_project)
    planner = ScriptedPlanner(
        [
            Action("apply_patch", {"path": "tests/test_calc.c", "diff": FIX}),
            Action("apply_patch", {"path": "src/calc.c", "diff": FIX}),
            done(True),
        ]
    )
    outcome = run_agent(toolbox, planner, tracer)
    first_result = planner.history[0][1]
    assert first_result.is_error and "protected" in first_result.content
    assert outcome.verified_fixed


def test_planner_error_stops_loop(toy_project):
    class Failing(ScriptedPlanner):
        def propose(self):
            raise PlannerError("api unavailable")

    toolbox, tracer, sink = make(toy_project)
    outcome = run_agent(toolbox, Failing([]), tracer)
    assert outcome.stop_reason == "planner_error"
    assert any(e["step"] == "error" and e["message"] == "api unavailable" for e in sink.events)


def test_briefing_contains_initial_failure(toy_project):
    toolbox, tracer, _ = make(toy_project)
    planner = ScriptedPlanner([done(False)])
    run_agent(toolbox, planner, tracer, AgentConfig(max_iters=3))
    assert "FAIL add_small" in planner.briefing
    assert "budget of 3 tool call(s)" in planner.briefing


def test_tool_specs_require_reasoning_fields(toy_project):
    specs = {s.name: s for s in planner_tool_specs(Toolbox(toy_project))}
    assert set(specs) == {"run_build", "run_tests", "read_file", "grep_source", "list_files", "apply_patch", DONE}
    for name, spec in specs.items():
        required = spec.input_schema["required"]
        assert "confidence" in required
        if name != DONE:
            assert required[:2] == ["belief", "confidence"]


def test_action_from_tool_input_strips_reasoning():
    action = Action.from_tool_input("read_file", {"belief": "b", "confidence": 0.3, "path": "x"}, "id1")
    assert action.args == {"path": "x"} and action.belief == "b" and action.confidence == 0.3
    final = Action.from_tool_input(DONE, {"root_cause": "rc", "summary": "s", "fixed": True, "confidence": 1})
    assert final.belief == "rc" and final.args["fixed"] is True


def test_usage_add():
    total = Usage()
    total.add(Usage(10, 5, 2, 1))
    total.add(Usage(1, 1, 0, 1))
    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens, total.model_calls) == (11, 6, 2, 2)
