import sys
import textwrap

import pytest

from autofix.config import ProjectConfig
from autofix.process import run_command, truncate_middle
from autofix.tools import TOOL_SPECS, Toolbox, validate_args


@pytest.fixture
def toolbox(toy_project):
    return Toolbox(toy_project)


def test_run_build_reports_failure_then_success(toolbox, toy_project):
    calc = toy_project.root / "src" / "calc.c"
    calc.write_text(calc.read_text().replace("return", "retrun"))
    failed = toolbox.call("run_build", {})
    assert failed.content.startswith("BUILD FAILED (exit code 1)")
    assert "src/calc.c:3:5: error" in failed.content
    assert failed.data["ok"] is False

    calc.write_text(calc.read_text().replace("retrun", "return"))
    assert toolbox.call("run_build", {}).content.startswith("BUILD SUCCEEDED")


def test_run_tests_lists_failures(toolbox):
    result = toolbox.call("run_tests", {})
    assert result.content.startswith("TESTS FAILED: 1 passed, 1 failed")
    assert "FAIL add_small at tests/test_calc.c:7: expected 5, got -1" in result.content
    assert result.data["failed"][0]["name"] == "add_small"
    assert result.data["all_passed"] is False


def test_run_tests_filter(toolbox):
    result = toolbox.call("run_tests", {"filter": "zero"})
    assert result.content.startswith("ALL TESTS PASSED: 1 passed, 0 failed")


def test_run_tests_stops_on_build_failure(toolbox, toy_project):
    calc = toy_project.root / "src" / "calc.c"
    calc.write_text(calc.read_text().replace("return", "retrun"))
    result = toolbox.call("run_tests", {})
    assert "tests were not run" in result.content
    assert result.data["build_ok"] is False


def test_read_file_numbers_and_ranges(toolbox):
    full = toolbox.call("read_file", {"path": "src/calc.c"})
    assert full.content.splitlines()[0] == "src/calc.c (lines 1-5 of 5)"
    assert "    4|     return a - b;" in full.content

    part = toolbox.call("read_file", {"path": "src/calc.c", "start_line": 3, "end_line": 4})
    assert part.content.splitlines()[1:] == ["    3| int add(int a, int b) {", "    4|     return a - b;"]

    assert toolbox.call("read_file", {"path": "src/calc.c", "start_line": 99}).is_error
    assert toolbox.call("read_file", {"path": "src/calc.c", "start_line": 4, "end_line": 2}).is_error


def test_read_file_refuses_escape_and_missing(toolbox):
    escape = toolbox.call("read_file", {"path": "../../etc/passwd"})
    assert escape.is_error and escape.content.startswith("refused:")
    assert escape.data["sandbox_violation"]
    missing = toolbox.call("read_file", {"path": "src/nope.c"})
    assert missing.is_error and "no such file" in missing.content


def test_grep_source(toolbox):
    result = toolbox.call("grep_source", {"pattern": r"return\s+a"})
    assert result.content == "src/calc.c:4:     return a - b;"
    assert toolbox.call("grep_source", {"pattern": "zzz"}).content == "no matches for 'zzz'"
    only_headers = toolbox.call("grep_source", {"pattern": "add", "glob": "**/*.h"})
    assert only_headers.content == "src/calc.h:1: int add(int a, int b);"


def test_grep_invalid_regex_falls_back_to_literal(toolbox):
    result = toolbox.call("grep_source", {"pattern": "add(int"})
    assert "invalid regex" in result.content
    assert "src/calc.c:3:" in result.content


def test_list_files(toolbox):
    assert toolbox.call("list_files", {}).content.splitlines() == [
        "src/calc.c",
        "src/calc.h",
        "tests/test_calc.c",
    ]


def test_apply_patch_fixes_and_records_changes(toolbox):
    diff = "@@ -4,1 +4,1 @@\n-    return a - b;\n+    return a + b;\n"
    result = toolbox.call("apply_patch", {"path": "src/calc.c", "diff": diff})
    assert result.content == "patched src/calc.c: 1 hunk(s), +1 -1"
    assert toolbox.call("run_tests", {}).data["all_passed"] is True
    assert toolbox.modified_files == ["src/calc.c"]
    changes = toolbox.changes()
    assert "--- a/src/calc.c" in changes and "+    return a + b;" in changes


def test_apply_patch_refuses_protected_files(toolbox):
    diff = "@@ -1,1 +1,1 @@\n-/* tests */\n+/* hacked */\n"
    result = toolbox.call("apply_patch", {"path": "tests/test_calc.c", "diff": diff})
    assert result.is_error and "protected" in result.content
    assert toolbox.modified_files == []


def test_apply_patch_reports_mismatch(toolbox):
    diff = "@@ -4,1 +4,1 @@\n-    return a * b;\n+    return a + b;\n"
    result = toolbox.call("apply_patch", {"path": "src/calc.c", "diff": diff})
    assert result.is_error and "does not apply" in result.content
    assert toolbox.modified_files == []


def test_call_validates_arguments(toolbox):
    assert "unknown tool" in toolbox.call("rm_rf", {}).content
    assert "missing required argument 'path'" in toolbox.call("read_file", {}).content
    assert "unexpected argument" in toolbox.call("run_build", {"force": True}).content
    assert "must be an integer" in toolbox.call("read_file", {"path": "a", "start_line": "1"}).content


def test_observations_are_redacted(tmp_path):
    root = tmp_path / "leaky"
    (root / "src").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools" / "build.py").write_text(
        textwrap.dedent(
            f"""
            import os
            print("using key AKIAIOSFODNN7EXAMPLE")
            print(os.path.abspath("src/main.c") + ":1:1: error: boom")
            raise SystemExit(1)
            """
        )
    )
    config = ProjectConfig.from_dict(
        root,
        {"build": {"command": [sys.executable, "tools/build.py"]}, "test": {"command": ["true"]}},
    )
    content = Toolbox(config).call("run_build", {}).content
    assert "AKIAIOSFODNN7EXAMPLE" not in content
    assert "src/main.c:1:1: error: boom" in content.replace("\\", "/")
    assert str(root) not in content


def test_every_spec_has_strict_object_schema():
    for spec in TOOL_SPECS:
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert set(spec.input_schema["required"]) <= set(spec.input_schema["properties"])


def test_validate_args_bounds():
    schema = {"type": "object", "properties": {"n": {"type": "number", "minimum": 0, "maximum": 1}}}
    assert validate_args(schema, {"n": 0.5}) is None
    assert "<=" in validate_args(schema, {"n": 2})
    assert "must be a number" in validate_args(schema, {"n": True})


def test_run_command_timeout_and_missing(tmp_path):
    slow = run_command([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path, timeout=0.5)
    assert slow.timed_out and slow.exit_code is None and not slow.ok
    missing = run_command(["definitely-not-a-real-program-xyz"], tmp_path, timeout=5)
    assert missing.exit_code == 127


def test_truncate_middle():
    text = "\n".join(f"line {i}" for i in range(1000))
    out = truncate_middle(text, 300)
    assert out.startswith("line 0") and out.endswith("line 999")
    assert "lines omitted" in out
    assert truncate_middle("short", 300) == "short"
