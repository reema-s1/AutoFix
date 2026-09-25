# AutoFix

AutoFix is an LLM agent that diagnoses and repairs failing C/C++ builds and
test suites. It works in a real **plan → act → observe** loop. Each turn the
model forms a hypothesis and picks one tool: build, run the tests, read a
file, search the source, or apply a patch. It then sees that tool's actual
output before deciding what to do next. It stops when the tests pass or when
it runs out of budget.

The harness trusts the tests, not the model. After every run the project is
rebuilt and re-tested, and the verified outcome is recorded next to what the
agent *claimed*. Claimed fixes that don't hold are measured and reported.

```
                     ┌──────────────────────────────┐
     failing build → │         AutoFix agent        │ → verified fix + diagnosis + trace
       or tests      │  plan → act → observe → ...  │
                     └──────────────┬───────────────┘
                                    │ one tool call per step
     ┌────────────┬───────────┬─────┴─────┬─────────────┬─────────────┐
     ▼            ▼           ▼           ▼             ▼             ▼
 run_build    run_tests   read_file   grep_source   list_files   apply_patch
     └────────────┴───────────┴─────┬─────┴─────────────┴─────────────┘
                                    │  sandboxed to the project, output redacted
                                    ▼
                        target C/C++ project (autofix.toml)
                                    │
                                    ▼
             trace: belief, confidence, action, result, tokens, verdict
                       (JSONL file, console, HTTP collector)
```

## Quick start

Requires Python 3.11+, a C/C++ toolchain for your project, and Anthropic API
credentials (`ANTHROPIC_API_KEY`, or a profile from `ant auth login`).

```bash
pip install -e ".[dev]"

# Reproduce one of the seeded bugs and let the agent fix it
autofix bugs list
autofix bugs materialize slidewin-window-wrap /tmp/slidewin
autofix fix /tmp/slidewin

# Replay the recorded run later
autofix replay runs/<run-id>.jsonl
```

`autofix fix` streams the agent's reasoning as it works:

```
=== run af-20260925-094641-e091 on calc (model=claude-opus-5, max_iters=8)
[0] observe  TESTS FAILED: 1 passed, 1 failed
[0] believe  add() subtracts instead of adding (confidence 0.80)
[0] act      apply_patch(path='src/calc.c', diff='@@ -4,1 +4,1 @@ ...')
[0] result   patched src/calc.c: 1 hunk(s), +1 -1
[1] believe  verify the fix (confidence 0.90)
[1] act      run_tests()
[1] result   ALL TESTS PASSED: 2 passed, 0 failed
[2] conclude fixed: add() in src/calc.c subtracts (confidence 0.95)
=== verified=True claimed=True iterations=2 tokens(in/out)=...
```

It exits `0` only if the harness verified the fix itself.

## Pointing it at a project

Add an `autofix.toml` at the project root:

```toml
[build]
command = ["cmake", "--build", "build"]
timeout = 300

[test]
command = ["ctest", "--test-dir", "build", "--output-on-failure"]
format = "exit-code"        # or "minitest" for per-test results (see below)
timeout = 120

[sandbox]
protected = ["tests/**", "CMakeLists.txt", "autofix.toml"]   # the agent may read but never modify these
source_globs = ["**/*.c", "**/*.h", "**/*.cpp", "**/*.hpp"]  # what grep_source and list_files see
```

Commands are argument lists and never run through a shell. With
`format = "minitest"`, the test runner prints one line per test:

```
RUN  ring_push_wraps
PASS ring_push_wraps
RUN  ring_pop_empty
FAIL ring_pop_empty (tests/test_ring.c:42): expected -1, got 0
SUMMARY passed=1 failed=1
```

The agent then sees exactly which tests failed and where. A `RUN` line with
no result means the process died inside that test, so crashes are attributed
to a test case. [fixtures/common/minitest.h](fixtures/common/minitest.h) is a
single-header C/C++ harness that emits this format.

## How it works

### The loop ([agent.py](src/autofix/agent.py))

1. The harness runs the build and the full test suite and briefs the planner
   with the result.
2. The planner proposes **one** action. Every tool schema requires a `belief`
   (current hypothesis) and a `confidence`, so the plan is structured data
   that gets traced at every step.
3. The harness validates and executes the action, then passes the real result
   back to the planner.
4. The loop repeats until the planner calls `declare_done` with a root cause,
   a summary, and whether it's fixed, or until the tool-call budget runs out.
   In that case the planner is asked for a final diagnosis without making
   more tool calls.
5. The harness rebuilds, re-runs every test, and records both the verified
   and the claimed outcome.

The loop itself doesn't depend on any particular model. `Planner` is an
interface. [ClaudePlanner](src/autofix/planners/claude.py) drives it with
Claude tool use: strict schemas, parallel tool use disabled so each model
call is exactly one step, adaptive thinking, and prompt caching over the
append-only conversation. [ScriptedPlanner](src/autofix/planners/scripted.py)
replays a fixed script, which is how the loop, tools and tracing are tested
without a model.

### Tools ([tools.py](src/autofix/tools.py))

| Tool | What it does |
|---|---|
| `run_build` | Runs the build and returns the exit code and output. Long output is truncated in the middle, because errors cluster at both ends. |
| `run_tests` | Rebuilds, runs the suite (optionally filtered), and returns per-test outcomes, failure locations, crash reasons, and raw output. |
| `read_file` | Returns a file with line numbers, paged in 400-line chunks. |
| `grep_source` | Searches source files by regex and returns `path:line: text`. An invalid regex falls back to a literal search. |
| `list_files` | Lists the project's source files. |
| `apply_patch` | Applies a unified diff to one file. |

Tools never raise on expected failures such as a missing file, a bad regex or
a patch that doesn't apply. They return an error result written for the
model, so it can correct itself on the next turn.

`apply_patch` ([patching.py](src/autofix/patching.py)) expects model-written
diffs. Their content is usually right, but their line numbers and hunk counts
often aren't. Each hunk is located by its context lines: the search starts
outward from the header's line hint, then falls back to matching that ignores
trailing whitespace. All hunks apply atomically, or none do.

### Guardrails

- **Sandbox** ([sandbox.py](src/autofix/sandbox.py)): every model-supplied
  path is resolved against the project root, following symlinks, and rejected
  if it escapes. Writes to protected globs are refused, so the agent can't
  "fix" a failure by editing the test. The evaluation also hashes the
  protected files before and after each run.
- **Redaction** ([redaction.py](src/autofix/redaction.py)): tool output is
  scrubbed before it reaches the model or any trace. That covers private keys,
  values of sensitive environment variables (`*_TOKEN`, `*KEY*`, ...),
  credential-shaped strings (cloud keys, API tokens, JWTs, bearer tokens,
  quoted `password = "..."` assignments, credentials in URLs), and absolute
  paths outside the project, which often reveal user names. Paths inside the
  project are rewritten as relative, so compiler locations stay usable. The
  [test suite](tests/test_redaction.py) pins down what must be removed *and*
  what must survive untouched, such as ordinary C code and compiler
  diagnostics.
- **Timeouts and bounded output** on every subprocess.

### Tracing ([tracing.py](src/autofix/tracing.py))

Every step is emitted as a JSON event that shares a `run_id`:

```json
{"run_id": "af-20260925-143012-7c1e", "seq": 4, "ts": "2026-09-25T14:30:19.412+00:00",
 "iter": 2, "step": "plan", "belief": "seq_in_window requires seq >= base, which breaks after wraparound",
 "confidence": 0.7, "tool": "read_file", "args": {"path": "src/window.c", "start_line": 1, "end_line": 40}}
```

The step types are `run_start`, `observation`, `plan`, `tool_result`,
`error`, `terminate` and `run_end` (verified vs. claimed outcome, token
usage). Events are redacted before they reach any sink. By default, traces go
to `runs/<run-id>.jsonl` and to the console. Setting `AUTOFIX_TRACE_URL`
(and optionally `AUTOFIX_TRACE_TOKEN`) also POSTs batches of events as
`{"events": [...]}` to an HTTP trace collector. That export runs on a
background thread, so a slow or unavailable collector never stalls the agent.

## Evaluation

[fixtures/](fixtures/) contains four small projects with thorough test
suites:

| Project | Language | What it covers |
|---|---|---|
| `ringbuf` | C | fixed-capacity circular buffer |
| `slidewin` | C | sliding-window ARQ: 8-bit wrapping sequence numbers, cumulative acks, selective-repeat receiver, RFC 1071 checksum |
| `strkit` | C | trimming, overflow-safe integer parsing, splitting, snprintf-style joining |
| `lrucache` | C++ | LRU cache built on `std::list` and `std::unordered_map` |

[fixtures/bugs/](fixtures/bugs/) holds 16 seeded bugs. Each one is a
declarative mutation of a clean project with a documented root cause:

| Category | Examples |
|---|---|
| build (4) | conflicting declaration types, missing `#include`, `static` vs. external linkage, `operator[]` in a `const` member |
| logic (9) | off-by-one capacity and ack bounds, sequence-number wraparound, odd-length checksum padding, LRU recency not refreshed |
| crash (1) | NULL slot dereference for frames outside the receive window |
| memory (2) | NUL terminator written one byte past the buffer, dangling iterator after eviction |

A test materializes every bug and checks that it produces its declared
symptom (build failure, test failure or crash), so the dataset can't rot
silently.

```bash
autofix eval                        # all bugs, budgets 1 and 8, LLM-judged diagnoses
autofix eval --only strkit --budgets 1 4 8 --jobs 4
autofix eval --judge keywords       # offline diagnosis scoring
```

Each run is scored independently of what the agent claims:

- **Fixed**: the project's own build and tests pass afterwards, and the
  protected files are byte-for-byte unchanged.
- **Diagnosed**: the stated root cause matches the known one. By default an
  LLM judge with structured output decides. `--judge keywords` checks
  keyword groups offline instead.
- **Hallucinated fix / honest give-up**: of the runs that didn't end with
  passing tests, how many claimed success anyway, and how many said they
  weren't fixed.
- **Iterations and tokens**.

**Ablation.** The default budgets `1 8` run the same catalog with a single
tool call (one shot: read the failure, make one change, stop) and with a full
loop. The difference between the two rows shows what iterating adds.
Results are written to `runs/eval-<timestamp>/`: `report.md`, `results.json`,
and one trace per run.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite runs offline. Model interactions are tested against a fake client.
Tests that compile the fixtures are marked `needs_cc` and skipped when no C
compiler is on `PATH`.

```
src/autofix/
  agent.py        plan-act-observe loop, action/outcome types, planner interface
  planners/       Claude tool-use planner, scripted planner
  tools.py        tool specs and sandboxed implementations
  patching.py     forgiving unified-diff applier
  sandbox.py      path confinement and write protection
  redaction.py    secret and path scrubbing
  process.py      subprocess execution, crash decoding
  testreport.py   test output parsing
  tracing.py      trace events, sinks, replay rendering
  bugs.py         seeded-bug catalog
  evaluation.py   scoring, judges, ablation report
  cli.py          command-line interface
fixtures/         evaluation projects and seeded bugs
tests/            test suite
```
