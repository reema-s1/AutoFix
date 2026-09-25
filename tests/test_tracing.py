import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from autofix.redaction import Redactor
from autofix.tracing import HttpSink, JsonlSink, MemorySink, Tracer, load_trace, new_run_id, render_trace


def test_emit_sequences_and_redacts():
    sink = MemorySink()
    tracer = Tracer(run_id="r1", sinks=[sink], redactor=Redactor(env={}))
    tracer.emit("run_start", project="demo")
    event = tracer.emit("tool_result", iter=0, summary="key AKIAIOSFODNN7EXAMPLE", skipped=None)
    assert event["seq"] == 2 and event["iter"] == 0 and event["run_id"] == "r1"
    assert "AKIA" not in event["summary"]
    assert "skipped" not in event
    assert sink.steps() == ["run_start", "tool_result"]


def test_jsonl_roundtrip_and_filter(tmp_path):
    path = tmp_path / "t.jsonl"
    for run in ("a", "b"):
        with Tracer(run_id=run, sinks=[JsonlSink(path)], redactor=Redactor(env={})) as tracer:
            tracer.emit("run_start")
            tracer.emit("plan", iter=0, belief="x", tool="run_build", args={}, confidence=0.5)
    assert len(load_trace(path)) == 4
    only_b = load_trace(path, run_id="b")
    assert [e["step"] for e in only_b] == ["run_start", "plan"]


def test_broken_sink_does_not_raise(capsys):
    class Boom:
        def write(self, event):
            raise RuntimeError("disk full")

        def close(self):
            pass

    good = MemorySink()
    tracer = Tracer(sinks=[Boom(), good], redactor=Redactor(env={}))
    tracer.emit("run_start")
    assert good.steps() == ["run_start"]
    assert "disk full" in capsys.readouterr().err


def test_http_sink_posts_batches():
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append(
                {"auth": self.headers.get("Authorization"), "body": json.loads(self.rfile.read(length))}
            )
            self.send_response(202)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        sink = HttpSink(f"http://127.0.0.1:{server.server_port}/ingest", token="t0k", batch_size=2)
        tracer = Tracer(sinks=[sink], redactor=Redactor(env={}))
        for i in range(3):
            tracer.emit("plan", iter=i)
        tracer.close()
    finally:
        server.shutdown()
    events = [e for r in received for e in r["body"]["events"]]
    assert [e["iter"] for e in events] == [0, 1, 2]
    assert all(r["auth"] == "Bearer t0k" for r in received)


def test_http_sink_failure_is_reported_once(capsys):
    sink = HttpSink("http://127.0.0.1:9/unreachable", timeout=0.5)
    tracer = Tracer(sinks=[sink], redactor=Redactor(env={}))
    tracer.emit("run_start")
    tracer.emit("run_end")
    tracer.close()
    assert capsys.readouterr().err.count("trace export") == 1


def test_render_trace():
    events = [
        {"run_id": "r", "step": "run_start", "project": "ring", "model": "m", "max_iters": 8},
        {"run_id": "r", "step": "observation", "iter": 0, "summary": "1 failed"},
        {"run_id": "r", "step": "plan", "iter": 0, "belief": "off by one", "tool": "read_file",
         "args": {"path": "src/ring.c"}, "confidence": 0.7},
        {"run_id": "r", "step": "tool_result", "iter": 0, "summary": "read 40 lines", "is_error": False},
        {"run_id": "r", "step": "terminate", "iter": 1, "reason": "declared_done", "summary": "fixed"},
        {"run_id": "r", "step": "run_end", "verified_fixed": True, "claimed_fixed": True, "iterations": 2,
         "input_tokens": 10, "output_tokens": 5},
    ]
    text = render_trace(events)
    assert "believe  off by one (confidence 0.70)" in text
    assert "act      read_file(path='src/ring.c')" in text
    assert "verified=True" in text


def test_new_run_id_is_unique():
    assert new_run_id() != new_run_id()
