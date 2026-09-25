"""ChatPlanner against a local fake server speaking Ollama and OpenAI wire formats."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from autofix import cli
from autofix.agent import DONE, AgentConfig, PlannerError, run_agent
from autofix.bugs import load_bugs
from autofix.evaluation import ChatJudge
from autofix.planners.chat import (
    ChatPlanner,
    ChatReply,
    OllamaBackend,
    OpenAICompatibleBackend,
    _inline_tool_call,
    _post_json,
)
from autofix.redaction import Redactor
from autofix.tools import ToolResult, Toolbox, ToolSpec
from autofix.tracing import MemorySink, Tracer

FIX = "@@ -4,1 +4,1 @@\n-    return a - b;\n+    return a + b;\n"
SPECS = [
    ToolSpec("read_file", "Read", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    ToolSpec(DONE, "Done", {"type": "object", "properties": {}, "required": []}),
]


class FakeServer:
    """Replies with queued JSON bodies (or (status, body, headers) tuples) and records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "user_agent": self.headers.get("User-Agent"),
                                       "body": json.loads(self.rfile.read(length))})
                reply = outer.replies.pop(0)
                status, body, headers = reply if isinstance(reply, tuple) else (200, reply, {})
                data = json.dumps(body).encode()
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def serve():
    servers = []

    def start(replies):
        server = FakeServer(replies)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def ollama_reply(name=None, arguments=None, content="", done_reason="stop"):
    message = {"role": "assistant", "content": content}
    if name:
        message["tool_calls"] = [{"function": {"name": name, "arguments": arguments or {}}}]
    return {"message": message, "done_reason": done_reason, "prompt_eval_count": 50, "eval_count": 10}


def openai_reply(name=None, arguments=None, content=None, call_id="call_abc"):
    message = {"role": "assistant", "content": content}
    if name:
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(arguments or {})}}]
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if name else "stop"}],
            "usage": {"prompt_tokens": 70, "completion_tokens": 7}}


def test_ollama_request_and_tool_roundtrip(serve):
    server = serve([ollama_reply("read_file", {"belief": "b", "confidence": 0.4, "path": "a.c"}),
                    ollama_reply(DONE, {"root_cause": "rc", "summary": "s", "fixed": False, "confidence": 0.2})])
    planner = ChatPlanner(OllamaBackend("qwen", base_url=server.url, num_ctx=8192))
    planner.begin("briefing", SPECS)
    action = planner.propose()
    planner.observe(action, ToolResult("contents", is_error=True))
    final = planner.propose()

    assert action.tool == "read_file" and action.args == {"path": "a.c"} and action.belief == "b"
    assert final.tool == DONE and final.args["root_cause"] == "rc"
    first = server.requests[0]
    assert first["path"] == "/api/chat"
    assert first["body"]["options"]["num_ctx"] == 8192 and first["body"]["stream"] is False
    assert first["body"]["tools"][0]["function"]["name"] == "read_file"
    assert [m["role"] for m in first["body"]["messages"]] == ["system", "user"]
    history = server.requests[1]["body"]["messages"]
    assert history[-1] == {"role": "tool", "tool_name": "read_file", "content": "ERROR: contents"}
    assert planner.usage.model_calls == 2 and planner.usage.input_tokens == 100
    assert planner.model == "ollama:qwen"


def test_openai_request_and_tool_roundtrip(serve):
    server = serve([openai_reply("read_file", {"belief": "b", "confidence": 0.5, "path": "a.c"}, call_id="c1"),
                    openai_reply(DONE, {"root_cause": "rc", "summary": "s", "fixed": True, "confidence": 1})])
    planner = ChatPlanner(OpenAICompatibleBackend("llama", base_url=server.url + "/v1", api_key="k3y"))
    planner.begin("briefing", SPECS)
    action = planner.propose()
    planner.observe(action, ToolResult("contents"))
    planner.propose()

    first = server.requests[0]
    assert first["path"] == "/v1/chat/completions" and first["auth"] == "Bearer k3y"
    assert first["user_agent"].startswith("autofix/")
    assert first["body"]["tool_choice"] == "auto"
    history = server.requests[1]["body"]["messages"]
    assert history[-2]["tool_calls"][0]["id"] == "c1"
    assert json.loads(history[-2]["tool_calls"][0]["function"]["arguments"])["path"] == "a.c"
    assert history[-1] == {"role": "tool", "tool_call_id": "c1", "content": "contents"}


def test_only_first_of_parallel_calls_is_kept(serve):
    reply = ollama_reply("read_file", {"path": "a.c"})
    reply["message"]["tool_calls"].append({"function": {"name": "read_file", "arguments": {"path": "b.c"}}})
    server = serve([reply])
    planner = ChatPlanner(OllamaBackend("m", base_url=server.url))
    planner.begin("b", SPECS)
    assert planner.propose().args == {"path": "a.c"}
    assert len(planner.messages[-1]["tool_calls"]) == 1


def test_inline_json_tool_call_is_recovered(serve):
    text = 'I will read it.\n```json\n{"name": "read_file", "arguments": {"path": "a.c", "belief": "x"}}\n```'
    server = serve([ollama_reply(content=text)])
    planner = ChatPlanner(OllamaBackend("m", base_url=server.url))
    planner.begin("b", SPECS)
    action = planner.propose()
    assert action.tool == "read_file" and action.args == {"path": "a.c"}
    planner.observe(action, ToolResult("ok"))  # the recovered call gets a matching result


def test_nudges_then_gives_up(serve):
    server = serve([ollama_reply(content="thinking...")] * 3)
    planner = ChatPlanner(OllamaBackend("m", base_url=server.url))
    planner.begin("b", SPECS)
    with pytest.raises(PlannerError, match="without calling a tool"):
        planner.propose()
    assert server.requests[1]["body"]["messages"][-1]["content"].startswith("Continue by calling")


def test_truncated_reply_is_an_error(serve):
    server = serve([ollama_reply(content="partial", done_reason="length")])
    planner = ChatPlanner(OllamaBackend("m", base_url=server.url))
    planner.begin("b", SPECS)
    with pytest.raises(PlannerError, match="cut off"):
        planner.propose()


def test_rate_limit_is_retried(serve):
    server = serve([(429, {"error": "slow down"}, {"Retry-After": "0"}), openai_reply(content="hi")])
    slept = []
    data = _post_json(server.url + "/v1/chat/completions", {}, {}, timeout=5, sleep=slept.append)
    assert data["choices"][0]["message"]["content"] == "hi" and slept == [0.0]


@pytest.mark.parametrize(
    "status, body, message",
    [(401, {"error": "bad key"}, "request refused .401.*bad key"), (404, {"error": "model 'x' not found"}, "model not found"),
     (400, {"error": "nope"}, "HTTP 400")],
)
def test_http_errors_become_planner_errors(serve, status, body, message):
    server = serve([(status, body, {})])
    with pytest.raises(PlannerError, match=message):
        _post_json(server.url, {}, {}, timeout=5)


def test_unreachable_server():
    with pytest.raises(PlannerError, match="could not reach"):
        _post_json("http://127.0.0.1:9/api/chat", {}, {}, timeout=1, max_retries=0)


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"name": "read_file", "parameters": {"path": "x"}}', ("read_file", {"path": "x"})),
        ('Call: {"tool": "read_file", "args": {"path": "y"}}', ("read_file", {"path": "y"})),
        ('{"name": "rm_rf", "arguments": {}}', None),
        ("no json here", None),
        ("{broken", None),
    ],
)
def test_inline_parser(content, expected):
    call = _inline_tool_call(content, {"read_file"})
    assert (None if call is None else (call.name, call.arguments)) == expected


def test_end_to_end_repairs_toy_project(serve, toy_project):
    server = serve([
        ollama_reply("read_file", {"belief": "add wrong", "confidence": 0.5, "path": "src/calc.c"}),
        ollama_reply("apply_patch", {"belief": "subtracts", "confidence": 0.9, "path": "src/calc.c", "diff": FIX}),
        ollama_reply("run_tests", {"belief": "verify", "confidence": 0.9}),
        ollama_reply(DONE, {"root_cause": "add() subtracts", "summary": "use +", "fixed": True, "confidence": 0.9}),
    ])
    outcome = run_agent(
        Toolbox(toy_project),
        ChatPlanner(OllamaBackend("qwen", base_url=server.url)),
        Tracer(sinks=[MemorySink()], redactor=Redactor(env={})),
        AgentConfig(max_iters=6),
    )
    assert outcome.verified_fixed and outcome.claimed_fixed and outcome.iterations == 3


def test_chat_judge_parses_json_and_falls_back():
    bug = next(b for b in load_bugs(cli.DEFAULT_FIXTURES / "bugs") if b.id == "ringbuf-peek-index")

    class Backend:
        name, model = "ollama", "m"

        def __init__(self, content):
            self.content = content

        def complete(self, messages, tools):
            return ChatReply(self.content)

    good = ChatJudge(Backend('Sure: {"correct": true, "rationale": "same defect"}')).judge(bug, "peek uses head")
    assert good.correct and good.method == "llm:ollama:m"
    fallback = ChatJudge(Backend("I think yes")).judge(bug, "ring_peek reads head instead of tail")
    assert fallback.correct and fallback.method == "keywords"


def test_cli_provider_configuration(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    parse = lambda *a: cli.build_parser().parse_args(["fix", ".", *a])  # noqa: E731

    ollama = cli._planner(parse("--provider", "ollama"))
    assert ollama.model == "ollama:qwen2.5-coder:7b"
    assert ollama.backend.base_url == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:9999")
    assert cli._planner(parse("--provider", "ollama")).backend.base_url == "http://0.0.0.0:9999"

    with pytest.raises(cli.ConfigError, match="GROQ_API_KEY"):
        cli._planner(parse("--provider", "groq"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    groq = cli._planner(parse("--provider", "groq"))
    assert groq.backend.base_url == "https://api.groq.com/openai/v1" and groq.backend.api_key == "gsk_test"

    with pytest.raises(cli.ConfigError, match="--model is required"):
        cli._planner(parse("--provider", "openai-compatible", "--base-url", "http://x/v1"))
    args = SimpleNamespace(judge="auto", provider="ollama")
    assert type(cli._judge(args)).__name__ == "KeywordJudge"


def test_rejected_tool_call_is_explained_and_retried(serve):
    rejected = (400, {"error": {"message": "Tool call validation failed: missing properties: 'path'",
                                "type": "invalid_request_error", "code": "tool_use_failed"}}, {})
    server = serve([rejected, openai_reply("read_file", {"path": "a.c"})])
    planner = ChatPlanner(OpenAICompatibleBackend("m", base_url=server.url))
    planner.begin("b", SPECS)
    assert planner.propose().args == {"path": "a.c"}
    retry_messages = server.requests[1]["body"]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "rejected before it ran: Tool call validation failed" in retry_messages[-1]["content"]


def test_repeated_rejections_give_up(serve):
    rejected = (400, {"error": {"message": "bad", "code": "tool_use_failed"}}, {})
    server = serve([rejected] * 3)
    planner = ChatPlanner(OpenAICompatibleBackend("m", base_url=server.url))
    planner.begin("b", SPECS)
    with pytest.raises(PlannerError, match="without calling a tool"):
        planner.propose()


def test_long_rate_limit_is_fatal_without_waiting(serve):
    from autofix.agent import RateLimited

    daily = (429, {"error": {"message": "tokens per day (TPD): Limit 200000", "code": "rate_limit_exceeded"}},
             {"Retry-After": "910"})
    server = serve([daily])
    slept = []
    with pytest.raises(RateLimited, match="tokens per day") as info:
        _post_json(server.url, {}, {}, timeout=5, sleep=slept.append)
    assert info.value.fatal and slept == []


def test_rate_limit_retries_exhausted_is_fatal(serve):
    from autofix.agent import RateLimited

    server = serve([(429, {"error": "slow"}, {"Retry-After": "0"})] * 3)
    with pytest.raises(RateLimited):
        _post_json(server.url, {}, {}, timeout=5, max_retries=2, sleep=lambda _: None)
