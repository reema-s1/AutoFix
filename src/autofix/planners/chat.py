"""A planner for open-weight and hosted chat models with function calling.

Two wire formats are supported:

* ``ollama``: Ollama's native ``/api/chat``, which (unlike its
  OpenAI-compatible endpoint) lets us set the context window. The default
  window is too small for build logs and source files.
* ``openai``: any OpenAI-compatible ``/chat/completions`` endpoint, such as
  Groq, OpenRouter, LM Studio, vLLM or llama.cpp's server.

Smaller models sometimes write a tool call as JSON in their reply text
instead of using the structured ``tool_calls`` field. Those replies are
recognised and treated as tool calls rather than wasted turns.

Only the standard library is used for HTTP, so no extra dependency is needed.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .. import __version__
from ..agent import DONE, Action, PlannerError, RateLimited, Usage
from ..tools import ToolResult, ToolSpec
from .prompts import BUDGET_EXHAUSTED, MAX_NUDGES, NUDGE, SYSTEM_PROMPT


class ToolCallRejected(PlannerError):
    """The provider refused the model's tool call (e.g. arguments failed schema validation)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    truncated: bool = False


class ChatBackend(Protocol):
    name: str
    model: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatReply: ...

    def assistant_message(self, reply: ChatReply, call: ToolCall | None) -> dict[str, Any]: ...

    def tool_message(self, call: ToolCall, content: str) -> dict[str, Any]: ...


# -- HTTP ---------------------------------------------------------------------

USER_AGENT = f"autofix/{__version__}"
# Per-minute limits clear quickly and are worth waiting out; daily quotas are not.
MAX_RATE_LIMIT_WAIT = 120.0


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    max_retries: int = 5,
    sleep=time.sleep,
) -> dict[str, Any]:
    """POST JSON, retrying rate limits and transient server errors with backoff."""
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            data=body,
            # Some API gateways reject urllib's default User-Agent outright.
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429:
                delay = _retry_delay(exc.headers.get("Retry-After"), attempt)
                if attempt >= max_retries or delay > MAX_RATE_LIMIT_WAIT:
                    raise RateLimited(f"rate limited by {url}: {_error_message(detail)}") from exc
                sleep(delay)
                continue
            if exc.code in (500, 502, 503, 504) and attempt < max_retries:
                sleep(min(_retry_delay(exc.headers.get("Retry-After"), attempt), MAX_RATE_LIMIT_WAIT))
                continue
            if exc.code == 400 and "tool_use_failed" in detail:
                raise ToolCallRejected(_error_message(detail)) from exc
            if exc.code == 404 and "model" in detail.lower():
                raise PlannerError(f"model not found at {url}: {detail}") from exc
            if exc.code in (401, 403):
                raise PlannerError(f"request refused ({exc.code}): {detail.strip()}; check the API key") from exc
            raise PlannerError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < max_retries and not isinstance(exc, TimeoutError):
                sleep(_retry_delay(None, attempt))
                continue
            raise PlannerError(f"could not reach {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PlannerError(f"invalid JSON from {url}") from exc
    raise PlannerError(f"giving up on {url} after {max_retries} retries")


def _error_message(detail: str) -> str:
    try:
        return str(json.loads(detail)["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return detail


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    try:
        if retry_after is not None:
            return float(retry_after)
    except ValueError:
        pass
    return min(2.0 ** attempt * 2, 60.0)


def _function_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]


# -- backends -----------------------------------------------------------------


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        num_ctx: int = 16_384,
        temperature: float = 0.2,
        timeout: float = 1_800,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatReply:
        data = _post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
            },
            headers={},
            timeout=self.timeout,
        )
        message = data.get("message") or {}
        calls = [
            ToolCall(f"call_{i}", c["function"]["name"], _parse_arguments(c["function"].get("arguments")))
            for i, c in enumerate(message.get("tool_calls") or [])
            if c.get("function", {}).get("name")
        ]
        prompt_tokens = data.get("prompt_eval_count") or 0
        return ChatReply(
            content=message.get("content") or "",
            tool_calls=calls,
            usage=Usage(prompt_tokens, data.get("eval_count") or 0, 0, 1),
            truncated=data.get("done_reason") == "length",
        )

    def assistant_message(self, reply: ChatReply, call: ToolCall | None) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if call is not None:
            message["tool_calls"] = [{"function": {"name": call.name, "arguments": call.arguments}}]
        return message

    def tool_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_name": call.name, "content": content}


class OpenAICompatibleBackend:
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8_192,
        timeout: float = 600,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatReply:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            headers=headers,
            timeout=self.timeout,
        )
        choices = data.get("choices") or []
        if not choices:
            raise PlannerError(f"no choices in response: {json.dumps(data)[:300]}")
        message = choices[0].get("message") or {}
        calls = [
            ToolCall(c.get("id") or f"call_{i}", c["function"]["name"], _parse_arguments(c["function"].get("arguments")))
            for i, c in enumerate(message.get("tool_calls") or [])
            if c.get("function", {}).get("name")
        ]
        usage = data.get("usage") or {}
        return ChatReply(
            content=message.get("content") or "",
            tool_calls=calls,
            usage=Usage(usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0, 0, 1),
            truncated=choices[0].get("finish_reason") == "length",
        )

    def assistant_message(self, reply: ChatReply, call: ToolCall | None) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.content or None}
        if call is not None:
            message["tool_calls"] = [
                {"id": call.id, "type": "function",
                 "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
            ]
        return message

    def tool_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call.id, "content": content}


# -- planner ------------------------------------------------------------------


class ChatPlanner:
    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend
        self.model = backend.model
        self.usage = Usage()
        self.messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []
        self._tool_names: set[str] = set()
        self._pending: ToolCall | None = None
        self._calls = 0

    def begin(self, briefing: str, tools: list[ToolSpec]) -> None:
        self._tools = _function_tools(tools)
        self._tool_names = {t.name for t in tools}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": briefing}]
        self._pending = None

    def propose(self) -> Action:
        for _ in range(MAX_NUDGES + 1):
            try:
                reply = self.backend.complete(self.messages, self._tools)
            except ToolCallRejected as exc:
                # The call never reached us, so there is nothing to answer; explain and retry.
                self.messages.append({"role": "user", "content": _REJECTED.format(reason=exc)})
                continue
            self.usage.add(reply.usage)
            call = reply.tool_calls[0] if reply.tool_calls else _inline_tool_call(reply.content, self._tool_names)
            if call is not None:
                self._calls += 1
                call.id = call.id if reply.tool_calls else f"call_inline_{self._calls}"
                # Record only the call we will execute, so every call in history has a result.
                self.messages.append(self.backend.assistant_message(reply, call))
                self._pending = call
                return Action.from_tool_input(call.name, call.arguments, call.id)
            if reply.truncated:
                raise PlannerError("model response was cut off before choosing an action")
            self.messages.append(self.backend.assistant_message(reply, None))
            self.messages.append({"role": "user", "content": NUDGE})
        raise PlannerError("model repeatedly ended its turn without calling a tool")

    def observe(self, action: Action, result: ToolResult) -> None:
        if self._pending is None or action.call_id != self._pending.id:
            raise PlannerError("observe() called for an action this planner did not propose")
        content = result.content or "(no output)"
        if result.is_error:
            content = f"ERROR: {content}"
        self.messages.append(self.backend.tool_message(self._pending, content))
        self._pending = None

    def conclude(self, reason: str) -> Action:
        self.messages.append({"role": "user", "content": BUDGET_EXHAUSTED})
        action = self.propose()
        if action.tool == DONE:
            return action
        return Action(
            DONE,
            {"root_cause": action.belief, "summary": reason, "fixed": False, "confidence": action.confidence or 0.0},
            action.belief,
            action.confidence,
        )


_REJECTED = (
    "Your last tool call was rejected before it ran: {reason}\n"
    "Call a tool again with arguments that match its schema exactly."
)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise PlannerError(f"tool arguments were not valid JSON: {raw[:200]}") from exc
        if isinstance(value, dict):
            return value
    raise PlannerError(f"tool arguments were not a JSON object: {raw!r:.200}")


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _inline_tool_call(content: str, tool_names: set[str]) -> ToolCall | None:
    """Recover a tool call written as JSON text, e.g. {"name": "read_file", "arguments": {...}}."""
    if not content or "{" not in content:
        return None
    candidates = [m.group(1) for m in _JSON_BLOCK.finditer(content)]
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        candidates.append(content[start : end + 1])
    for text in candidates:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        name = value.get("name") or value.get("tool") or value.get("function")
        args = value.get("arguments", value.get("parameters", value.get("args", {})))
        if isinstance(name, str) and name in tool_names:
            try:
                return ToolCall("", name, _parse_arguments(args))
            except PlannerError:
                continue
    return None
