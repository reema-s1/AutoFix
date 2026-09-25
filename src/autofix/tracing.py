"""Structured, replayable traces of every agent step.

Each run emits a sequence of JSON events sharing a ``run_id``::

    {"run_id": "af-20260925-143012-7c1e", "seq": 4, "ts": "...", "iter": 2,
     "step": "plan", "belief": "count is compared with > instead of >=",
     "tool": "read_file", "args": {"path": "src/ring.c"}, "confidence": 0.6}

Step types: ``run_start``, ``observation`` (the harness's initial build/test
run), ``plan`` (what the model believed and chose), ``tool_result``,
``model_usage``, ``error``, ``terminate`` and ``run_end``.

Events are redacted before they reach any sink. Sinks are pluggable: a JSONL
file (the default, replayable with ``autofix replay``), an HTTP exporter for
an external trace collector, and an in-memory sink for tests.
"""

from __future__ import annotations

import json
import queue
import secrets
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .redaction import Redactor


def new_run_id(prefix: str = "af") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"


class TraceSink(Protocol):
    def write(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def write(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def steps(self) -> list[str]:
        return [e["step"] for e in self.events]


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class HttpSink:
    """POSTs batches of events as ``{"events": [...]}`` to a collector endpoint.

    Export happens on a background thread so a slow or unavailable collector
    never stalls the agent; failures are reported once on stderr and the
    remaining events are dropped rather than retried indefinitely.
    """

    def __init__(
        self,
        url: str,
        token: str | None = None,
        batch_size: int = 20,
        timeout: float = 5.0,
    ) -> None:
        self.url = url
        self.token = token
        self.batch_size = batch_size
        self.timeout = timeout
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._failed = False
        self._thread = threading.Thread(target=self._run, name="autofix-trace-export", daemon=True)
        self._thread.start()

    def write(self, event: dict[str, Any]) -> None:
        if not self._failed:
            self._queue.put(event)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=self.timeout * 2)

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        while True:
            item = self._queue.get()
            if item is not None:
                batch.append(item)
            if batch and (item is None or len(batch) >= self.batch_size or self._queue.empty()):
                self._post(batch)
                batch = []
            if item is None:
                return

    def _post(self, batch: list[dict[str, Any]]) -> None:
        if self._failed:
            return
        body = json.dumps({"events": batch}, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._failed = True
            print(f"autofix: trace export to {self.url} failed ({exc}); continuing without it", file=sys.stderr)


class Tracer:
    def __init__(
        self,
        run_id: str | None = None,
        sinks: Iterable[TraceSink] = (),
        redactor: Redactor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or new_run_id()
        self.sinks = list(sinks)
        self.redactor = redactor or Redactor()
        self.metadata = dict(metadata or {})
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, step: str, iter: int | None = None, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event: dict[str, Any] = {
                "run_id": self.run_id,
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "step": step,
            }
            if iter is not None:
                event["iter"] = iter
            event.update({k: v for k, v in fields.items() if v is not None})
            event = self.redactor.redact_obj(event)
            for sink in self.sinks:
                try:
                    sink.write(event)
                except Exception as exc:  # a broken sink must never take the agent down
                    print(f"autofix: trace sink {type(sink).__name__} failed: {exc}", file=sys.stderr)
            return event

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_trace(path: str | Path, run_id: str | None = None) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                event = json.loads(line)
                if run_id is None or event.get("run_id") == run_id:
                    events.append(event)
    return events


def render_trace(events: list[dict[str, Any]], width: int = 100) -> str:
    """Human-readable replay of a run: beliefs, actions and what came back."""
    out: list[str] = []
    for e in events:
        step = e.get("step")
        it = f"[{e['iter']}]" if "iter" in e else "   "
        if step == "run_start":
            out.append(f"=== run {e['run_id']} on {e.get('project', '?')} (model={e.get('model', '?')}, max_iters={e.get('max_iters')})")
        elif step == "observation":
            out.append(f"{it} observe  {e.get('summary', '')}")
        elif step == "plan":
            conf = e.get("confidence")
            conf_s = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
            out.append(f"{it} believe  {_clip(e.get('belief', ''), width)}{conf_s}")
            out.append(f"{it} act      {e.get('tool')}({_clip(_fmt_args(e.get('args', {})), width)})")
        elif step == "tool_result":
            flag = "!" if e.get("is_error") else " "
            out.append(f"{it} result{flag}  {_clip(e.get('summary', ''), width)}")
        elif step == "terminate":
            out.append(f"{it} stop     {e.get('reason')}: {_clip(e.get('summary', ''), width)}")
        elif step == "run_end":
            out.append(
                f"=== verified={e.get('verified_fixed')} claimed={e.get('claimed_fixed')} "
                f"iterations={e.get('iterations')} tokens(in/out)={e.get('input_tokens')}/{e.get('output_tokens')}"
            )
        elif step == "error":
            out.append(f"{it} ERROR    {e.get('message')}")
    return "\n".join(out)


def _fmt_args(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        text = json.dumps(value) if not isinstance(value, str) else repr(value)
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."
