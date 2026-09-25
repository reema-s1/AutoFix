"""The agent's tools: build, test, read, search, list and patch a project.

Every tool is sandboxed to the project root and every observation is
redacted before it is returned. Tools never raise for expected failures
(missing file, bad regex, patch that does not apply); they return a
:class:`ToolResult` with ``is_error=True`` and a message written for the
model, so it can correct itself on the next iteration.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .patching import PatchError, apply_unified_diff, patch_target
from .process import CommandResult, run_command, truncate_middle
from .redaction import Redactor
from .sandbox import DEFAULT_IGNORED_DIRS, Sandbox, SandboxViolation
from .testreport import PARSERS, TestReport

MAX_READ_LINES = 400
MAX_GREP_RESULTS = 60
MAX_GREP_LINE = 200
MAX_LIST_FILES = 300


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    content: str  # model-facing text, already redacted
    is_error: bool = False
    data: dict[str, Any] = field(default_factory=dict)  # structured facts for traces/evaluation


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "run_build",
        "Run the project's configured build command. Returns the exit code and the "
        "compiler/linker output (long output is truncated in the middle).",
        _schema({}),
    ),
    ToolSpec(
        "run_tests",
        "Rebuild the project, then run its test suite. Returns per-test pass/fail, failure "
        "locations and messages, crash information, and the raw runner output.",
        _schema(
            {
                "filter": {
                    "type": "string",
                    "description": "Optional substring; only tests whose name contains it are run.",
                }
            }
        ),
    ),
    ToolSpec(
        "read_file",
        f"Read a project file with line numbers. Reads at most {MAX_READ_LINES} lines per call; "
        "use start_line/end_line (1-based, inclusive) to page through larger files.",
        _schema(
            {
                "path": {"type": "string", "description": "Project-relative path, e.g. src/ring.c"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            required=("path",),
        ),
    ),
    ToolSpec(
        "grep_source",
        "Search project source files with a regular expression (Python syntax). "
        "Returns matching lines as path:line: text.",
        _schema(
            {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "glob": {
                    "type": "string",
                    "description": "Optional file glob to restrict the search, e.g. src/**/*.c",
                },
            },
            required=("pattern",),
        ),
    ),
    ToolSpec(
        "list_files",
        "List project source files (build output and VCS directories are skipped).",
        _schema({"glob": {"type": "string", "description": "Optional glob, e.g. **/*.h"}}),
    ),
    ToolSpec(
        "apply_patch",
        "Apply a unified diff to ONE project file. Include a few unchanged context lines "
        "around each change; the context is used to locate the hunk, so copy it exactly "
        "from the current file. Test files and build configuration are read-only.",
        _schema(
            {
                "path": {
                    "type": "string",
                    "description": "Project-relative path of the file to change. May be omitted "
                    "if the diff names the file in its headers.",
                },
                "diff": {
                    "type": "string",
                    "description": "Unified diff with one or more '@@ -a,b +c,d @@' hunks.",
                },
            },
            required=("diff",),
        ),
    ),
)

TOOL_NAMES = frozenset(spec.name for spec in TOOL_SPECS)


class Toolbox:
    def __init__(
        self,
        config: ProjectConfig,
        redactor: Redactor | None = None,
        output_limit: int = 12_000,
    ) -> None:
        self.config = config
        self.sandbox = Sandbox(
            config.root,
            protected=config.protected,
            ignored_dirs=DEFAULT_IGNORED_DIRS | set(config.extra_ignored_dirs),
        )
        self.redactor = redactor or Redactor(project_root=config.root)
        self.output_limit = output_limit
        self._originals: dict[Path, str | None] = {}
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "run_build": self.run_build,
            "run_tests": self.run_tests,
            "read_file": self.read_file,
            "grep_source": self.grep_source,
            "list_files": self.list_files,
            "apply_patch": self.apply_patch,
        }

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return TOOL_SPECS

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Validate arguments and dispatch. Never raises for tool-level failures."""
        spec = next((s for s in TOOL_SPECS if s.name == name), None)
        if spec is None:
            return ToolResult(f"unknown tool {name!r}; available: {sorted(TOOL_NAMES)}", is_error=True)
        problem = validate_args(spec.input_schema, args)
        if problem:
            return ToolResult(f"invalid arguments for {name}: {problem}", is_error=True)
        try:
            return self._handlers[name](**args)
        except SandboxViolation as exc:
            return ToolResult(f"refused: {exc}", is_error=True, data={"sandbox_violation": True})
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError) as exc:
            return ToolResult(self.redactor.redact(str(exc)), is_error=True)

    # -- build & test -------------------------------------------------------

    def run_build(self) -> ToolResult:
        result = self._build()
        status = "BUILD SUCCEEDED" if result.ok else _failure_label("BUILD FAILED", result)
        content = f"{status} ({result.duration_s:.1f}s)\n{self._output(result)}".rstrip()
        return ToolResult(
            content,
            data={"exit_code": result.exit_code, "ok": result.ok, "timed_out": result.timed_out},
        )

    def run_tests(self, filter: str | None = None) -> ToolResult:
        build = self._build()
        if not build.ok:
            label = _failure_label("BUILD FAILED", build)
            return ToolResult(
                f"{label}; tests were not run.\n{self._output(build)}".rstrip(),
                data={"build_ok": False, "exit_code": build.exit_code},
            )

        argv = self.config.test.argv
        if filter and self.config.test_format == "minitest":
            argv = (*argv, filter)
        result = run_command(argv, self.config.root, self.config.test.timeout)
        report = PARSERS[self.config.test_format](result)

        lines = [("ALL TESTS PASSED" if report.all_passed else "TESTS FAILED") + f": {report.summary_line()}"]
        if filter and self.config.test_format != "minitest":
            lines.append(f"(note: this project's test runner does not support filtering; ran all tests)")
        for case in report.failed:
            where = f" at {case.location}" if case.location else ""
            detail = f": {case.message}" if case.message else ""
            lines.append(f"  {case.status.upper()} {case.name}{where}{detail}")
        lines.append("--- runner output ---")
        lines.append(self._output(result))
        return ToolResult(
            self.redactor.redact("\n".join(lines)).rstrip(),
            data={"build_ok": True, **self.redactor.redact_obj(report.to_dict())},
        )

    def test_report(self) -> TestReport | None:
        """Build and run the full suite, returning the parsed report (None if the build fails)."""
        if not self._build().ok:
            return None
        result = run_command(self.config.test.argv, self.config.root, self.config.test.timeout)
        return PARSERS[self.config.test_format](result)

    def _build(self) -> CommandResult:
        return run_command(self.config.build.argv, self.config.root, self.config.build.timeout)

    def _output(self, result: CommandResult) -> str:
        return truncate_middle(self.redactor.redact(result.output), self.output_limit)

    # -- reading & searching ------------------------------------------------

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        target = self.sandbox.check_readable(path)
        text = _read_text(target)
        lines = text.splitlines()
        total = len(lines)
        start = max(start_line or 1, 1)
        end = min(end_line or total, total, start + MAX_READ_LINES - 1)
        if total and start > total:
            return ToolResult(f"{self.sandbox.relative(target)} has only {total} lines", is_error=True)
        if end_line is not None and end_line < start:
            return ToolResult("end_line must be >= start_line", is_error=True)

        numbered = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, end + 1))
        rel = self.sandbox.relative(target)
        header = f"{rel} (lines {start}-{end} of {total})" if total else f"{rel} (empty file)"
        if end < total and end_line is None:
            header += f" - truncated; request start_line={end + 1} to continue"
        return ToolResult(
            self.redactor.redact(f"{header}\n{numbered}"),
            data={"path": rel, "start_line": start, "end_line": end, "total_lines": total},
        )

    def grep_source(self, pattern: str, glob: str | None = None) -> ToolResult:
        note = ""
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            regex = re.compile(re.escape(pattern))
            note = f"(invalid regex: {exc}; searched for the literal text instead)\n"

        globs = (glob,) if glob else self.config.source_globs
        hits: list[str] = []
        truncated = False
        for path in self.sandbox.iter_files(globs):
            try:
                text = _read_text(path)
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    if len(hits) >= MAX_GREP_RESULTS:
                        truncated = True
                        break
                    shown = line if len(line) <= MAX_GREP_LINE else line[:MAX_GREP_LINE] + "..."
                    hits.append(f"{self.sandbox.relative(path)}:{number}: {shown}")
            if truncated:
                break

        if not hits:
            return ToolResult(f"{note}no matches for {pattern!r}", data={"matches": 0})
        tail = f"\n... (stopped after {MAX_GREP_RESULTS} matches; narrow the pattern or glob)" if truncated else ""
        return ToolResult(
            self.redactor.redact(note + "\n".join(hits) + tail),
            data={"matches": len(hits), "truncated": truncated},
        )

    def list_files(self, glob: str | None = None) -> ToolResult:
        globs = (glob,) if glob else self.config.source_globs
        files = [self.sandbox.relative(p) for p in self.sandbox.iter_files(globs)]
        shown = files[:MAX_LIST_FILES]
        content = "\n".join(shown) if shown else "no matching files"
        if len(files) > len(shown):
            content += f"\n... ({len(files) - len(shown)} more)"
        return ToolResult(self.redactor.redact(content), data={"count": len(files)})

    # -- editing ------------------------------------------------------------

    def apply_patch(self, diff: str, path: str | None = None) -> ToolResult:
        # An explicit path wins: models often write loose headers such as "a/ring.c".
        path = path or patch_target(diff)
        if path is None:
            return ToolResult(
                "patch not applied: no file given; pass `path` or name the file in the diff headers",
                is_error=True,
            )
        target = self.sandbox.check_writable(path)
        original = _read_text(target) if target.exists() else None
        try:
            result = apply_unified_diff(original, diff)
        except PatchError as exc:
            return ToolResult(self.redactor.redact(f"patch not applied: {exc}"), is_error=True)

        self._originals.setdefault(target, original)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(result.new_text)

        rel = self.sandbox.relative(target)
        fuzz = " (located by context; line numbers in the diff were off)" if result.fuzzy else ""
        return ToolResult(
            f"patched {rel}: {result.hunks_applied} hunk(s), +{result.lines_added} -{result.lines_removed}{fuzz}",
            data={
                "path": rel,
                "hunks": result.hunks_applied,
                "added": result.lines_added,
                "removed": result.lines_removed,
                "fuzzy": result.fuzzy,
            },
        )

    def changes(self) -> str:
        """Unified diff of every file modified through :meth:`apply_patch`."""
        chunks = []
        for path, original in sorted(self._originals.items()):
            current = _read_text(path) if path.exists() else ""
            rel = self.sandbox.relative(path)
            chunks.extend(
                difflib.unified_diff(
                    (original or "").splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{rel}" if original is not None else "/dev/null",
                    tofile=f"b/{rel}",
                )
            )
        return self.redactor.redact("".join(chunks))

    @property
    def modified_files(self) -> list[str]:
        return sorted(self.sandbox.relative(p) for p in self._originals)


def validate_args(schema: dict[str, Any], args: Any) -> str | None:
    """Minimal JSON-schema check covering the shapes used by :data:`TOOL_SPECS`."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    properties = schema.get("properties", {})
    for name in schema.get("required", ()):
        if name not in args:
            return f"missing required argument {name!r}"
    for name, value in args.items():
        if name not in properties:
            if schema.get("additionalProperties") is False:
                return f"unexpected argument {name!r}"
            continue
        prop = properties[name]
        expected = prop.get("type")
        if value is None and name not in schema.get("required", ()):
            continue
        if expected == "string" and not isinstance(value, str):
            return f"{name} must be a string"
        if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return f"{name} must be an integer"
        if expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"{name} must be a number"
        if expected == "boolean" and not isinstance(value, bool):
            return f"{name} must be a boolean"
        if "minimum" in prop and isinstance(value, (int, float)) and value < prop["minimum"]:
            return f"{name} must be >= {prop['minimum']}"
        if "maximum" in prop and isinstance(value, (int, float)) and value > prop["maximum"]:
            return f"{name} must be <= {prop['maximum']}"
    return None


def _failure_label(label: str, result: CommandResult) -> str:
    if result.timed_out:
        return f"{label} (timed out)"
    if result.crash_reason:
        return f"{label} (crashed: {result.crash_reason})"
    return f"{label} (exit code {result.exit_code})"


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise UnicodeDecodeError("utf-8", data[:1], 0, 1, "binary file")
    return data.decode("utf-8", errors="replace")
