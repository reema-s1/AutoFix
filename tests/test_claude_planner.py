"""ClaudePlanner against a fake client: conversation shape, not model quality."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from autofix.agent import DONE, Action, AgentConfig, PlannerError, planner_tool_specs, run_agent
from autofix.planners.claude import ClaudePlanner, to_api_tool
from autofix.redaction import Redactor
from autofix.tools import ToolResult, Toolbox, ToolSpec
from autofix.tracing import MemorySink, Tracer


def tool_use(name, input, id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id)


def text(value):
    return SimpleNamespace(type="text", text=value)


def response(*content, stop_reason="tool_use", input_tokens=100, output_tokens=20):
    usage = SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_input_tokens=50, cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=list(content), stop_reason=stop_reason, usage=usage)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        # Snapshot the messages: the planner keeps appending to the same list.
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_client(responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


SPECS = [
    ToolSpec("read_file", "Read", {"type": "object", "properties": {
        "belief": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "path": {"type": "string"}}, "required": ["belief", "confidence", "path"], "additionalProperties": False}),
]


def test_request_shape_and_action_parsing():
    client, messages = fake_client(
        [response(text("Looking."), tool_use("read_file", {"belief": "b", "confidence": 0.4, "path": "a.c"}))]
    )
    planner = ClaudePlanner(client=client)
    planner.begin("briefing", SPECS)
    action = planner.propose()

    assert action == Action("read_file", {"path": "a.c"}, "b", 0.4, "tu_1")
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["messages"] == [{"role": "user", "content": "briefing"}]
    assert call["tools"][0]["strict"] is True
    assert planner.usage.model_calls == 1 and planner.usage.cache_read_tokens == 50


def test_observe_appends_matching_tool_result():
    client, messages = fake_client(
        [
            response(tool_use("read_file", {"belief": "b", "confidence": 0.4, "path": "a.c"}, id="tu_9")),
            response(tool_use(DONE, {"root_cause": "rc", "summary": "s", "fixed": False, "confidence": 0.2}, id="tu_10")),
        ]
    )
    planner = ClaudePlanner(client=client)
    planner.begin("briefing", SPECS)
    action = planner.propose()
    planner.observe(action, ToolResult("file contents", is_error=False))
    final = planner.propose()

    assert final.tool == DONE and final.belief == "rc"
    sent = messages.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_9", "content": "file contents", "is_error": False}
    ]


def test_observe_rejects_foreign_action():
    client, _ = fake_client([response(tool_use("read_file", {"belief": "", "confidence": 0, "path": "a"}))])
    planner = ClaudePlanner(client=client)
    planner.begin("b", SPECS)
    planner.propose()
    with pytest.raises(PlannerError):
        planner.observe(Action("read_file", {}, call_id="other"), ToolResult("x"))


def test_nudges_when_model_ends_turn_without_tool():
    client, messages = fake_client(
        [
            response(text("I think it's line 4."), stop_reason="end_turn"),
            response(tool_use(DONE, {"root_cause": "rc", "summary": "s", "fixed": False, "confidence": 0.1})),
        ]
    )
    planner = ClaudePlanner(client=client)
    planner.begin("b", SPECS)
    assert planner.propose().tool == DONE
    assert messages.calls[1]["messages"][-1]["content"].startswith("Continue by calling exactly one tool")


def test_gives_up_after_repeated_toolless_turns():
    client, _ = fake_client([response(text("hmm"), stop_reason="end_turn")] * 3)
    planner = ClaudePlanner(client=client)
    planner.begin("b", SPECS)
    with pytest.raises(PlannerError, match="without calling a tool"):
        planner.propose()


def test_refusal_becomes_planner_error():
    client, _ = fake_client([response(stop_reason="refusal")])
    planner = ClaudePlanner(client=client)
    planner.begin("b", SPECS)
    with pytest.raises(PlannerError, match="declined"):
        planner.propose()


def test_api_errors_become_planner_errors():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(request=request)
    client, _ = fake_client([error])
    planner = ClaudePlanner(client=client)
    planner.begin("b", SPECS)
    with pytest.raises(PlannerError, match="could not reach"):
        planner.propose()


def test_conclude_forces_declare_done():
    client, messages = fake_client(
        [
            response(tool_use("read_file", {"belief": "b", "confidence": 0.4, "path": "a.c"}, id="t1")),
            response(tool_use("read_file", {"belief": "still looking", "confidence": 0.3, "path": "b.c"}, id="t2")),
        ]
    )
    planner = ClaudePlanner(client=client)
    planner.begin("b", SPECS)
    action = planner.propose()
    planner.observe(action, ToolResult("contents"))
    final = planner.conclude("budget")

    last_user = messages.calls[1]["messages"][-1]
    assert last_user["content"][-1]["text"].startswith("The tool-call budget is exhausted")
    assert final.tool == DONE and final.args["fixed"] is False and final.args["root_cause"] == "still looking"


def test_to_api_tool_moves_numeric_bounds_into_description():
    tool = to_api_tool(SPECS[0])
    confidence = tool["input_schema"]["properties"]["confidence"]
    assert "minimum" not in confidence and "maximum" not in confidence
    assert "range 0 to 1" in confidence["description"]
    assert SPECS[0].input_schema["properties"]["confidence"]["minimum"] == 0  # original untouched


def test_end_to_end_with_fake_model(toy_project):
    fix = "@@ -4,1 +4,1 @@\n-    return a - b;\n+    return a + b;\n"
    client, _ = fake_client(
        [
            response(tool_use("read_file", {"belief": "add() is wrong", "confidence": 0.5, "path": "src/calc.c"}, id="1")),
            response(tool_use("apply_patch", {"belief": "subtracts", "confidence": 0.9, "path": "src/calc.c", "diff": fix}, id="2")),
            response(tool_use("run_tests", {"belief": "verify", "confidence": 0.9}, id="3")),
            response(tool_use(DONE, {"root_cause": "add() in src/calc.c subtracts", "summary": "use +", "fixed": True, "confidence": 0.95}, id="4")),
        ]
    )
    sink = MemorySink()
    outcome = run_agent(
        Toolbox(toy_project),
        ClaudePlanner(client=client),
        Tracer(sinks=[sink], redactor=Redactor(env={})),
        AgentConfig(max_iters=6),
    )
    assert outcome.verified_fixed and outcome.claimed_fixed and outcome.iterations == 3
    assert outcome.usage.model_calls == 4
    assert sink.events[0]["model"] == "claude-opus-5"
