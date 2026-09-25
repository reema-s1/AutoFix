import json
import sys
from pathlib import Path

import pytest

from autofix import cli
from autofix.agent import DONE, Action
from autofix.planners import ScriptedPlanner

FIX = "@@ -4,1 +4,1 @@\n-    return a - b;\n+    return a + b;\n"


@pytest.fixture
def toy_on_disk(toy_project):
    root = toy_project.root
    (root / "autofix.toml").write_text(
        "[build]\n"
        f"command = [{json.dumps(sys.executable)}, \"tools/build.py\"]\n"
        "[test]\n"
        f"command = [{json.dumps(sys.executable)}, \"tools/test.py\"]\n"
        "[sandbox]\nprotected = [\"tests/**\", \"tools/**\", \"autofix.toml\"]\n",
        encoding="utf-8",
    )
    return root


def scripted(*actions):
    return lambda args: ScriptedPlanner(list(actions))


def test_fix_repairs_and_writes_trace(toy_on_disk, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_planner",
        scripted(
            Action("apply_patch", {"path": "src/calc.c", "diff": FIX}, "subtracts", 0.8),
            Action("run_tests", {}, "verify", 0.9),
            Action(DONE, {"root_cause": "add() subtracts", "summary": "use +", "fixed": True, "confidence": 0.9}),
        ),
    )
    trace = tmp_path / "trace.jsonl"
    code = cli.main(["fix", str(toy_on_disk), "--trace-file", str(trace)])
    out, err = capsys.readouterr()

    assert code == 0
    assert "verified:   FIXED (agent claimed: fixed)" in out
    assert "+    return a + b;" in out
    assert "believe  subtracts (confidence 0.80)" in err  # live console trace
    assert json.loads(trace.read_text().splitlines()[-1])["verified_fixed"] is True


def test_fix_reports_hallucinated_fix(toy_on_disk, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_planner",
        scripted(Action(DONE, {"root_cause": "?", "summary": "", "fixed": True, "confidence": 0.9})),
    )
    code = cli.main(["fix", str(toy_on_disk), "--quiet", "--trace-file", str(tmp_path / "t.jsonl")])
    out = capsys.readouterr().out
    assert code == 1
    assert "NOT FIXED" in out and "warning:" in out


def test_fix_rejects_missing_config(tmp_path, capsys):
    assert cli.main(["fix", str(tmp_path)]) == 2
    assert "no autofix.toml" in capsys.readouterr().err


def test_fix_rejects_zero_budget(toy_on_disk, capsys):
    assert cli.main(["fix", str(toy_on_disk), "--max-iters", "0"]) == 2


def test_replay(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        json.dumps({"run_id": "r1", "step": "plan", "iter": 0, "belief": "b", "tool": "run_build", "args": {}})
        + "\n"
        + json.dumps({"run_id": "r2", "step": "plan", "iter": 0, "belief": "other", "tool": "x", "args": {}})
        + "\n"
    )
    assert cli.main(["replay", str(trace), "--run-id", "r1"]) == 0
    out = capsys.readouterr().out
    assert "believe  b" in out and "other" not in out
    assert cli.main(["replay", str(trace), "--run-id", "nope"]) == 1


def test_bugs_list(capsys):
    assert cli.main(["bugs", "list"]) == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 16
    assert "ringbuf-full-check" in out


def test_bugs_materialize(tmp_path, capsys):
    dest = tmp_path / "broken"
    assert cli.main(["bugs", "materialize", "ringbuf-peek-index", str(dest)]) == 0
    assert "r->buf[r->head];\n    return 0;" in (dest / "src" / "ring.c").read_text()
    assert cli.main(["bugs", "materialize", "ringbuf", str(dest)]) == 2
    assert "unknown bug id" in capsys.readouterr().err


def test_eval_with_keyword_judge(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_planner",
        scripted(Action(DONE, {"root_cause": "no idea", "summary": "", "fixed": False, "confidence": 0.1})),
    )
    out_dir = tmp_path / "eval"
    code = cli.main(
        ["eval", "--only", "strkit-missing-ctype", "--budgets", "1", "--judge", "keywords", "--out", str(out_dir)]
    )
    assert code == 0
    report = (out_dir / "report.md").read_text()
    assert "| strkit-missing-ctype | build | 1 | gave_up_honestly | no |" in report
