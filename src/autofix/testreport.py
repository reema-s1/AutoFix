"""Parsing test-runner output into per-test outcomes.

The ``minitest`` format is a deliberately simple line protocol that any
C/C++ harness can emit (see ``fixtures/common/minitest.h``)::

    RUN  ring_push_wraps
    PASS ring_push_wraps
    RUN  ring_pop_empty
    FAIL ring_pop_empty (tests/test_ring.c:42): expected -1, got 0
    SUMMARY passed=1 failed=1

A ``RUN`` with no matching ``PASS``/``FAIL`` means the process died inside
that test, which is how crashes are attributed to a test case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .process import CommandResult

_RUN = re.compile(r"^RUN\s+(\S+)\s*$")
_PASS = re.compile(r"^PASS\s+(\S+)\s*$")
_FAIL = re.compile(r"^FAIL\s+(\S+)\s*(?:\(([^)]*)\))?\s*:?\s*(.*)$")
_SUMMARY = re.compile(r"^SUMMARY\s+passed=(\d+)\s+failed=(\d+)")


@dataclass
class TestCase:
    __test__ = False  # not a pytest test class

    name: str
    status: str  # "pass" | "fail" | "crash"
    location: str | None = None
    message: str | None = None


@dataclass
class TestReport:
    __test__ = False

    cases: list[TestCase] = field(default_factory=list)
    exit_code: int | None = None
    timed_out: bool = False
    crash_reason: str | None = None
    summary_seen: bool = False

    @property
    def passed(self) -> list[TestCase]:
        return [c for c in self.cases if c.status == "pass"]

    @property
    def failed(self) -> list[TestCase]:
        return [c for c in self.cases if c.status != "pass"]

    @property
    def all_passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.failed

    def summary_line(self) -> str:
        if self.timed_out:
            return "TIMEOUT: test run exceeded its time limit"
        head = f"{len(self.passed)} passed, {len(self.failed)} failed"
        if self.crash_reason:
            head += f"; process crashed: {self.crash_reason}"
        elif self.exit_code not in (0, None) and not self.failed:
            head += f"; runner exited with code {self.exit_code}"
        return head

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "passed": [c.name for c in self.passed],
            "failed": [
                {"name": c.name, "status": c.status, "location": c.location, "message": c.message}
                for c in self.failed
            ],
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "crash_reason": self.crash_reason,
        }


def parse_minitest(result: CommandResult) -> TestReport:
    report = TestReport(
        exit_code=result.exit_code, timed_out=result.timed_out, crash_reason=result.crash_reason
    )
    running: str | None = None
    for line in result.output.splitlines():
        line = line.strip()
        if m := _RUN.match(line):
            running = m.group(1)
        elif m := _PASS.match(line):
            report.cases.append(TestCase(m.group(1), "pass"))
            running = None
        elif m := _FAIL.match(line):
            report.cases.append(TestCase(m.group(1), "fail", m.group(2), m.group(3) or None))
            running = None
        elif _SUMMARY.match(line):
            report.summary_seen = True

    if running is not None:
        reason = "timed out" if result.timed_out else (result.crash_reason or "process exited mid-test")
        report.cases.append(TestCase(running, "crash", message=reason))
    return report


def parse_exit_code(result: CommandResult) -> TestReport:
    report = TestReport(
        exit_code=result.exit_code, timed_out=result.timed_out, crash_reason=result.crash_reason
    )
    status = "pass" if result.ok else ("crash" if result.crash_reason or result.timed_out else "fail")
    report.cases.append(TestCase("<suite>", status, message=None if result.ok else result.crash_reason))
    return report


PARSERS = {"minitest": parse_minitest, "exit-code": parse_exit_code}
