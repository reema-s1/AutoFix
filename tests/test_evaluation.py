import difflib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofix.agent import DONE, Action
from autofix.bugs import load_bugs
from autofix.evaluation import (
    BugResult,
    ClaudeJudge,
    EvalConfig,
    KeywordJudge,
    classify,
    evaluate,
    render_markdown,
    summarize,
    write_results,
)
from autofix.planners import ScriptedPlanner

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
BUGS = {b.id: b for b in load_bugs(FIXTURES / "bugs")}


def oracle_planner(bug):
    """Scripted planner that repairs the bug by reverting its mutation."""
    mutation = bug.mutations[0]

    def fix(history):
        _, read = history[-1]
        numbered = [line.split("| ", 1)[1] if "| " in line else "" for line in read.content.splitlines()[1:]]
        broken = "\n".join(numbered) + "\n"
        repaired = broken.replace(mutation.replace, mutation.find, 1)
        diff = "".join(
            difflib.unified_diff(broken.splitlines(True), repaired.splitlines(True), "a", "b", n=2)
        )
        return Action("apply_patch", {"path": mutation.file, "diff": diff}, "revert the defect", 0.9)

    return ScriptedPlanner(
        [
            Action("read_file", {"path": mutation.file}, "inspect", 0.5),
            fix,
            Action("run_tests", {}, "verify", 0.9),
            Action(DONE, {"root_cause": bug.root_cause, "summary": "reverted", "fixed": True, "confidence": 0.9}),
        ]
    )


@pytest.mark.parametrize(
    "verified, claimed, expected",
    [
        (True, True, "fixed"),
        (True, False, "fixed_unclaimed"),
        (True, None, "fixed_unclaimed"),
        (False, True, "hallucinated"),
        (False, False, "gave_up_honestly"),
        (False, None, "no_verdict"),
    ],
)
def test_classify(verified, claimed, expected):
    assert classify(verified, claimed) == expected


def test_keyword_judge():
    bug = BUGS["slidewin-ack-bound"]
    judge = KeywordJudge()
    assert judge.judge(bug, "sender_on_ack uses >= where > is needed for in_flight").correct
    assert not judge.judge(bug, "the ack test fails").correct


def test_claude_judge_uses_structured_output():
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parsed_output=kwargs["output_format"](correct=True, rationale="same defect"))

    client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    verdict = ClaudeJudge(client=client).judge(BUGS["ringbuf-peek-index"], "peek reads head not tail")
    assert verdict.correct and verdict.method == "llm:claude-opus-5"
    assert "<known_root_cause>" in calls[0]["messages"][0]["content"]


def test_claude_judge_rejects_empty_diagnosis_without_calling():
    client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **_: pytest.fail("should not be called")))
    assert not ClaudeJudge(client=client).judge(BUGS["ringbuf-peek-index"], "  ").correct


def result(**overrides):
    base = dict(
        bug_id="b", project="p", category="logic", symptom="test_failure", max_iters=8, run_id="r",
        outcome="fixed", stop_reason="declared_done", iterations=3, claimed_fixed=True, verified_fixed=True,
        tests_untouched=True, diagnosis_correct=True, judge_method="keywords", judge_rationale="",
        root_cause="rc", modified_files=["src/a.c"], input_tokens=100, output_tokens=50, duration_s=1.0,
    )
    base.update(overrides)
    return BugResult(**base)


def test_summarize_and_render():
    results = [
        result(bug_id="a", iterations=2),
        result(bug_id="b", iterations=4, category="build"),
        result(bug_id="c", outcome="hallucinated", verified_fixed=False, diagnosis_correct=False),
        result(bug_id="d", outcome="gave_up_honestly", verified_fixed=False, claimed_fixed=False),
        result(bug_id="a", max_iters=1, outcome="gave_up_honestly", verified_fixed=False, claimed_fixed=False,
               iterations=1),
    ]
    one, eight = summarize(results)
    assert (one.max_iters, one.runs, one.fix_rate) == (1, 1, 0.0)
    assert eight.fix_rate == 0.5
    assert eight.diagnosis_rate == 0.75
    assert eight.hallucination_rate == 0.5 and eight.honest_giveup_rate == 0.5
    assert eight.mean_iterations_when_fixed == 3.0
    assert eight.by_category["build"]["fix_rate"] == 1.0
    assert eight.outcomes == {"fixed": 2, "hallucinated": 1, "gave_up_honestly": 1}

    markdown = render_markdown(results, [one, eight])
    assert "| 8 | 4 | 50% | 75% | 50% | 50% | 3.0 |" in markdown
    assert "| 1 | 1 | 0% |" in markdown


def test_tampered_tests_do_not_count_as_fixed():
    [summary] = summarize([result(tests_untouched=False)])
    assert summary.fix_rate == 0.0
    assert "(tests modified!)" in render_markdown([result(tests_untouched=False)], [summary])


def test_write_results(tmp_path):
    results = [result()]
    json_path, md_path = write_results(results, summarize(results), tmp_path)
    data = json.loads(json_path.read_text())
    assert data["results"][0]["bug_id"] == "b" and data["summaries"][0]["fix_rate"] == 1.0
    assert md_path.read_text().startswith("# AutoFix evaluation")


@pytest.mark.needs_cc
def test_end_to_end_with_oracle_planner(tmp_path):
    bugs = [BUGS["ringbuf-full-check"], BUGS["strkit-missing-ctype"]]
    planners = iter(oracle_planner(b) for b in bugs)
    config = EvalConfig(fixtures_dir=FIXTURES, work_dir=tmp_path, budgets=(8,))
    results = evaluate(bugs, config, lambda: next(planners), KeywordJudge())

    for r in results:
        assert r.outcome == "fixed", (r.bug_id, r.outcome, r.error)
        assert r.tests_untouched and r.diagnosis_correct and r.iterations == 3
    assert (tmp_path / "traces" / "ringbuf-full-check-k8.jsonl").is_file()


@pytest.mark.needs_cc
def test_budget_of_one_cannot_verify(tmp_path):
    bug = BUGS["ringbuf-peek-index"]
    config = EvalConfig(fixtures_dir=FIXTURES, work_dir=tmp_path, budgets=(1,))
    [r] = evaluate([bug], config, lambda: oracle_planner(bug), KeywordJudge())
    # Only the read happens before the budget is spent; the planner concludes without a fix.
    assert r.iterations == 1 and not r.verified_fixed
    assert r.outcome == "hallucinated"  # the oracle script claims fixed=True regardless
