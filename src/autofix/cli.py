"""Command-line interface.

    autofix fix PROJECT_DIR            diagnose and repair a failing project
    autofix eval                       run the seeded-bug evaluation
    autofix replay TRACE.jsonl         replay a recorded run
    autofix bugs list|materialize      inspect or reproduce seeded bugs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .agent import AgentConfig, PlannerError, run_agent
from .bugs import BugSpecError, load_bugs, materialize
from .config import ConfigError, ProjectConfig
from .tools import Toolbox
from .tracing import HttpSink, JsonlSink, Tracer, TraceSink, load_trace, new_run_id, render_trace

DEFAULT_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ConsoleSink:
    """Prints each trace event as it happens, in replay format."""

    def __init__(self, stream=None) -> None:
        self.stream = stream

    def write(self, event: dict[str, Any]) -> None:
        line = render_trace([event])
        if line:
            print(line, file=self.stream or sys.stderr, flush=True)

    def close(self) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ConfigError, BugSpecError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"autofix: {exc}", file=sys.stderr)
        return 2
    except PlannerError as exc:  # fatal model errors, e.g. an exhausted quota
        print(f"autofix: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("autofix: interrupted", file=sys.stderr)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autofix", description="Diagnose and repair failing C/C++ builds and tests with an LLM agent."
    )
    parser.add_argument("--version", action="version", version=f"autofix {__version__}")
    sub = parser.add_subparsers(required=True, metavar="COMMAND")

    fix = sub.add_parser("fix", help="diagnose and repair a failing C/C++ project")
    fix.add_argument("project", type=Path, help="project directory containing autofix.toml")
    _add_model_args(fix)
    fix.add_argument("--max-iters", type=int, default=8, help="tool-call budget (default: 8)")
    fix.add_argument("--trace-file", type=Path, help="JSONL trace path (default: runs/<run-id>.jsonl)")
    fix.add_argument("--quiet", action="store_true", help="do not print steps as they happen")
    fix.set_defaults(handler=cmd_fix)

    ev = sub.add_parser("eval", help="evaluate against the seeded-bug catalog")
    _add_model_args(ev)
    ev.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES, help="fixtures directory")
    ev.add_argument("--only", nargs="+", metavar="ID", help="bug ids or project names to include")
    ev.add_argument("--budgets", type=int, nargs="+", default=[1, 8],
                    help="tool-call budgets to compare (default: 1 8)")
    ev.add_argument("--jobs", type=int, default=1, help="bugs to run in parallel")
    ev.add_argument("--judge", choices=("auto", "llm", "keywords"), default="auto",
                    help="how to score diagnoses: an LLM judge or keyword groups "
                         "(default: auto = llm for claude, keywords otherwise)")
    out = ev.add_mutually_exclusive_group()
    out.add_argument("--out", type=Path, help="output directory (default: runs/eval-<timestamp>)")
    out.add_argument("--resume", type=Path, metavar="DIR",
                     help="continue an interrupted evaluation in DIR with its original settings")
    ev.set_defaults(handler=cmd_eval)

    replay = sub.add_parser("replay", help="replay a recorded trace")
    replay.add_argument("trace", type=Path)
    replay.add_argument("--run-id", help="only show this run")
    replay.set_defaults(handler=cmd_replay)

    bugs = sub.add_parser("bugs", help="inspect or reproduce seeded bugs")
    bugs.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    bugs_sub = bugs.add_subparsers(required=True, metavar="ACTION")
    bugs_list = bugs_sub.add_parser("list", help="list seeded bugs")
    bugs_list.set_defaults(handler=cmd_bugs_list)
    bugs_mat = bugs_sub.add_parser("materialize", help="write a broken copy of a fixture")
    bugs_mat.add_argument("bug_id")
    bugs_mat.add_argument("dest", type=Path)
    bugs_mat.set_defaults(handler=cmd_bugs_materialize)
    return parser


PROVIDERS = {
    # name: (default model, default base URL, env var holding the API key)
    "claude": ("claude-opus-5", None, None),
    "ollama": ("qwen2.5-coder:7b", "http://127.0.0.1:11434", None),
    "groq": ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai-compatible": (None, None, "AUTOFIX_API_KEY"),
}


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        default=os.environ.get("AUTOFIX_PROVIDER", "claude"),
                        help="model provider (default: $AUTOFIX_PROVIDER or claude)")
    parser.add_argument("--model", default=os.environ.get("AUTOFIX_MODEL"),
                        help="model id (default: $AUTOFIX_MODEL or the provider's default)")
    parser.add_argument("--base-url", default=os.environ.get("AUTOFIX_BASE_URL"),
                        help="API base URL for ollama / openai-compatible providers")
    parser.add_argument("--effort", choices=EFFORTS, default="high", help="Claude reasoning effort (default: high)")
    parser.add_argument("--num-ctx", type=int, default=16_384,
                        help="Ollama context window in tokens (default: 16384)")


def _model(args: argparse.Namespace) -> str:
    model = args.model or PROVIDERS[args.provider][0]
    if not model:
        raise ConfigError(f"--model is required for provider {args.provider!r}")
    return model


def _chat_backend(args: argparse.Namespace):
    from .planners.chat import OllamaBackend, OpenAICompatibleBackend

    _, default_url, key_env = PROVIDERS[args.provider]
    if args.provider == "ollama":
        base_url = args.base_url or os.environ.get("OLLAMA_HOST") or default_url
        if "://" not in base_url:  # OLLAMA_HOST is often given as host:port
            base_url = f"http://{base_url}"
        return OllamaBackend(_model(args), base_url=base_url, num_ctx=args.num_ctx)
    base_url = args.base_url or default_url
    if not base_url:
        raise ConfigError("--base-url is required for provider 'openai-compatible'")
    api_key = os.environ.get(key_env) if key_env else None
    if args.provider == "groq" and not api_key:
        raise ConfigError("set GROQ_API_KEY (free key at https://console.groq.com/keys)")
    return OpenAICompatibleBackend(_model(args), base_url=base_url, api_key=api_key)


def _planner(args: argparse.Namespace):
    if args.provider == "claude":
        from .planners import ClaudePlanner

        return ClaudePlanner(model=_model(args), effort=args.effort)
    from .planners.chat import ChatPlanner

    return ChatPlanner(_chat_backend(args))


def _judge(args: argparse.Namespace):
    from .evaluation import ChatJudge, ClaudeJudge, KeywordJudge

    mode = args.judge
    if mode == "auto":
        mode = "llm" if args.provider == "claude" else "keywords"
    if mode == "keywords":
        return KeywordJudge()
    if args.provider == "claude":
        return ClaudeJudge(model=_model(args))
    return ChatJudge(_chat_backend(args))


def _remote_sinks() -> list[TraceSink]:
    url = os.environ.get("AUTOFIX_TRACE_URL")
    return [HttpSink(url, token=os.environ.get("AUTOFIX_TRACE_TOKEN"))] if url else []


def cmd_fix(args: argparse.Namespace) -> int:
    if args.max_iters < 1:
        raise ConfigError("--max-iters must be at least 1")
    project = ProjectConfig.load(args.project)
    toolbox = Toolbox(project)
    run_id = new_run_id()
    trace_path = args.trace_file or Path("runs") / f"{run_id}.jsonl"
    sinks: list[TraceSink] = [JsonlSink(trace_path), *_remote_sinks()]
    if not args.quiet:
        sinks.append(ConsoleSink())

    with Tracer(run_id=run_id, sinks=sinks, redactor=toolbox.redactor) as tracer:
        outcome = run_agent(toolbox, _planner(args), tracer, AgentConfig(max_iters=args.max_iters))

    print()
    print(f"run:        {outcome.run_id}")
    print(f"stopped:    {outcome.stop_reason} after {outcome.iterations} tool call(s)")
    print(f"verified:   {'FIXED' if outcome.verified_fixed else 'NOT FIXED'}"
          f" (agent claimed: {_claim(outcome.claimed_fixed)})")
    if outcome.hallucinated_fix:
        print("warning:    the agent claimed a fix that the test suite does not confirm")
    if outcome.root_cause:
        print(f"root cause: {outcome.root_cause}")
    if outcome.changes:
        print("\nchanges:\n" + outcome.changes)
    print(f"trace:      {trace_path}")
    return 0 if outcome.verified_fixed else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import EvalAborted, EvalConfig, evaluate, load_results, summarize, write_results

    previous = []
    if args.resume:
        out_dir = args.resume
        settings_path = out_dir / EVAL_SETTINGS
        if not settings_path.is_file():
            raise ConfigError(f"{out_dir} is not an evaluation directory (no {EVAL_SETTINGS})")
        for key, value in json.loads(settings_path.read_text(encoding="utf-8")).items():
            setattr(args, key, value)
        previous = load_results(out_dir)
    else:
        out_dir = args.out or Path("runs") / f"eval-{datetime.now():%Y%m%d-%H%M%S}"

    if any(b < 1 for b in args.budgets):
        raise ConfigError("--budgets must all be at least 1")
    bugs = load_bugs(args.fixtures / "bugs", set(args.only) if args.only else None)
    config = EvalConfig(
        fixtures_dir=args.fixtures, work_dir=out_dir, budgets=tuple(sorted(set(args.budgets))), jobs=args.jobs
    )
    judge = _judge(args)

    def planner_factory():
        return _planner(args)

    planner_factory()  # fail fast on configuration errors before any work starts
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        settings = {k: getattr(args, k) for k in _EVAL_SETTING_KEYS}
        settings["model"] = _model(args)
        (out_dir / EVAL_SETTINGS).write_text(json.dumps(settings, indent=2), encoding="utf-8")

    total = len(bugs) * len(config.budgets)
    done = sum((r.bug_id, r.max_iters) in {(b.id, k) for b in bugs for k in config.budgets}
               for r in previous if r.outcome != "error")

    def progress(result) -> None:
        nonlocal done
        done += 1
        diag = "diagnosed" if result.diagnosis_correct else "misdiagnosed"
        error = f" [error: {result.error}]" if result.error else ""
        print(f"[{done}/{total}] {result.bug_id} (budget {result.max_iters}): {result.outcome}, {diag}, "
              f"{result.iterations} call(s){error}", file=sys.stderr, flush=True)

    resumed = f", resuming with {done} already done" if done else ""
    print(f"evaluating {len(bugs)} bug(s) x budgets {list(config.budgets)} -> {out_dir}{resumed}", file=sys.stderr)
    try:
        results = evaluate(bugs, config, planner_factory, judge, on_result=progress, previous=previous)
    except EvalAborted as exc:
        print(f"autofix: stopping evaluation, the model could not be used: {exc}", file=sys.stderr)
        if exc.results:
            _, md_path = write_results(exc.results, summarize(exc.results), out_dir)
            print(f"partial report ({len(exc.results)}/{total} runs): {md_path}", file=sys.stderr)
        print(f"continue later with: autofix eval --resume {out_dir}", file=sys.stderr)
        return 1
    summaries = summarize(results)
    json_path, md_path = write_results(results, summaries, out_dir)
    print(md_path.read_text(encoding="utf-8"))
    print(f"results: {json_path}\nreport:  {md_path}", file=sys.stderr)
    return 0


EVAL_SETTINGS = "settings.json"
_EVAL_SETTING_KEYS = ("provider", "model", "base_url", "effort", "num_ctx", "only", "budgets", "judge")


def cmd_replay(args: argparse.Namespace) -> int:
    events = load_trace(args.trace, run_id=args.run_id)
    if not events:
        print("no events found", file=sys.stderr)
        return 1
    print(render_trace(events, width=140))
    return 0


def cmd_bugs_list(args: argparse.Namespace) -> int:
    bugs = load_bugs(args.fixtures / "bugs")
    width = max(len(b.id) for b in bugs)
    for bug in bugs:
        print(f"{bug.id:<{width}}  {bug.category:<7} {bug.symptom:<14} {bug.root_cause}")
    return 0


def cmd_bugs_materialize(args: argparse.Namespace) -> int:
    bug = next((b for b in load_bugs(args.fixtures / "bugs") if b.id == args.bug_id), None)
    if bug is None:
        raise BugSpecError(f"unknown bug id {args.bug_id!r}; see `autofix bugs list`")
    dest = materialize(bug, args.fixtures, args.dest)
    print(f"wrote broken copy of {bug.project} to {dest}")
    print(f"try: autofix fix {dest}")
    return 0


def _claim(value: bool | None) -> str:
    return {True: "fixed", False: "not fixed", None: "no verdict"}[value]


if __name__ == "__main__":
    sys.exit(main())
