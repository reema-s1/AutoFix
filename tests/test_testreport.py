from autofix.process import CommandResult, describe_crash
from autofix.testreport import parse_exit_code, parse_minitest


def result(output: str, exit_code: int | None = 0, timed_out: bool = False) -> CommandResult:
    return CommandResult(("t",), exit_code, output, 0.1, timed_out=timed_out)


def test_parses_pass_and_fail():
    out = """\
RUN  push_one
PASS push_one
RUN  pop_empty
FAIL pop_empty (tests/test_ring.c:42): expected -1, got 0
SUMMARY passed=1 failed=1
"""
    report = parse_minitest(result(out, exit_code=1))
    assert [c.name for c in report.passed] == ["push_one"]
    [failure] = report.failed
    assert failure.name == "pop_empty"
    assert failure.location == "tests/test_ring.c:42"
    assert failure.message == "expected -1, got 0"
    assert report.summary_seen
    assert not report.all_passed
    assert report.summary_line() == "1 passed, 1 failed"


def test_crash_is_attributed_to_running_test():
    out = "RUN  a\nPASS a\nRUN  b\n"
    report = parse_minitest(result(out, exit_code=0xC0000005))
    [crash] = report.failed
    assert crash.name == "b" and crash.status == "crash"
    assert "access violation" in crash.message
    assert "crashed" in report.summary_line()


def test_timeout_attributed_to_running_test():
    report = parse_minitest(result("RUN  spin\n", exit_code=None, timed_out=True))
    assert report.failed[0].message == "timed out"
    assert report.summary_line().startswith("TIMEOUT")


def test_nonzero_exit_without_failures_is_not_success():
    report = parse_minitest(result("RUN  a\nPASS a\nSUMMARY passed=1 failed=0\n", exit_code=3))
    assert not report.all_passed
    assert "exited with code 3" in report.summary_line()


def test_exit_code_format():
    assert parse_exit_code(result("", 0)).all_passed
    failed = parse_exit_code(result("boom", 1))
    assert failed.failed[0].status == "fail"
    assert parse_exit_code(result("", -11)).failed[0].status == "crash"


def test_describe_crash():
    assert describe_crash(0) is None
    assert describe_crash(1) is None
    assert "SIGSEGV" in describe_crash(-11)
    assert "SIGSEGV" in describe_crash(139)
    assert "stack overflow" in describe_crash(0xC00000FD)
    assert "access violation" in describe_crash(-1073741819)  # signed form of 0xC0000005
