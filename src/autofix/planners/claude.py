"""A planner backed by Claude tool use.

The conversation is append-only: the briefing, then for each iteration the
assistant turn (thinking + exactly one tool call) followed by a user turn
carrying that tool's real result. Parallel tool use is disabled so each
model call is one plan -> act step and the iteration budget is exact.
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..agent import DONE, Action, PlannerError, RateLimited, Usage
from ..tools import ToolResult, ToolSpec
from .prompts import BUDGET_EXHAUSTED, MAX_NUDGES, NUDGE, SYSTEM_PROMPT

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_UNSUPPORTED_STRICT_KEYWORDS = ("minimum", "maximum")


class ClaudePlanner:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 16_000,
        client: Any | None = None,
        server_side_fallback: bool = True,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()
        self.server_side_fallback = server_side_fallback
        self.usage = Usage()
        self.messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []
        self._pending_tool_use: str | None = None

    # -- Planner protocol -----------------------------------------------------

    def begin(self, briefing: str, tools: list[ToolSpec]) -> None:
        self._tools = [to_api_tool(spec) for spec in tools]
        self.messages = [{"role": "user", "content": briefing}]
        self._pending_tool_use = None

    def propose(self) -> Action:
        for _ in range(MAX_NUDGES + 1):
            response = self._create()
            self.messages.append({"role": "assistant", "content": response.content})
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            if tool_use is not None:
                self._pending_tool_use = tool_use.id
                return Action.from_tool_input(tool_use.name, _as_dict(tool_use.input), tool_use.id)
            if response.stop_reason == "max_tokens":
                raise PlannerError("model response hit max_tokens before choosing an action")
            self.messages.append({"role": "user", "content": NUDGE})
        raise PlannerError("model repeatedly ended its turn without calling a tool")

    def observe(self, action: Action, result: ToolResult) -> None:
        if action.call_id is None or action.call_id != self._pending_tool_use:
            raise PlannerError("observe() called for an action this planner did not propose")
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": action.call_id,
                        "content": result.content or "(no output)",
                        "is_error": result.is_error,
                    }
                ],
            }
        )
        self._pending_tool_use = None

    def conclude(self, reason: str) -> Action:
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "user" and isinstance(last["content"], list):
            last["content"].append({"type": "text", "text": BUDGET_EXHAUSTED})
        else:
            self.messages.append({"role": "user", "content": BUDGET_EXHAUSTED})
        action = self.propose()
        if action.tool == DONE:
            return action
        # The model tried to keep working; record its belief as an unverified diagnosis.
        return Action(
            DONE,
            {"root_cause": action.belief, "summary": reason, "fixed": False, "confidence": action.confidence or 0.0},
            action.belief,
            action.confidence,
        )

    # -- API ------------------------------------------------------------------

    def _create(self):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=self._tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            messages=self.messages,
        )
        if self.server_side_fallback:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        try:
            response = self.client.beta.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            raise PlannerError(f"request rejected by the API: {exc.message}") from exc
        except anthropic.AuthenticationError as exc:
            raise PlannerError("authentication failed; set ANTHROPIC_API_KEY or run `ant auth login`") from exc
        except anthropic.RateLimitError as exc:
            raise RateLimited("rate limited after retries; try again later") from exc
        except anthropic.APIStatusError as exc:
            raise PlannerError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise PlannerError(f"could not reach the API: {exc}") from exc

        self._record_usage(response)
        if response.stop_reason == "refusal":
            raise PlannerError("the model declined this request")
        return response

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage.add(
            Usage(
                input_tokens=(usage.input_tokens or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                model_calls=1,
            )
        )


def to_api_tool(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec to a strict Messages API tool definition.

    Numeric range keywords are not supported by strict tool schemas, so they
    are moved into the property description instead (and enforced locally).
    """
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": _strict_schema(spec.input_schema),
        "strict": True,
    }


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    out = dict(schema)
    if "properties" in out:
        out["properties"] = {name: _strict_property(prop) for name, prop in out["properties"].items()}
    return out


def _strict_property(prop: dict[str, Any]) -> dict[str, Any]:
    prop = dict(prop)
    bounds = {k: prop.pop(k) for k in _UNSUPPORTED_STRICT_KEYWORDS if k in prop}
    if bounds:
        rng = f"range {bounds.get('minimum', '-inf')} to {bounds.get('maximum', 'inf')}"
        prop["description"] = f"{prop.get('description', '').rstrip()} ({rng})".strip()
    return prop


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise PlannerError(f"tool input was not a JSON object: {value!r}")
