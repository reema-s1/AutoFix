"""A forgiving unified-diff applier for model-authored patches.

Language models are good at writing the *content* of a hunk but unreliable
at line numbers and hunk-header counts. This applier therefore:

* ignores the ``+c,d`` counts and derives hunk extent from line prefixes,
* treats the ``-a`` start line only as a hint, searching outward from it for
  the hunk's context + removed lines,
* falls back to a trailing-whitespace-insensitive match,
* applies all hunks atomically: either every hunk lands or the file is
  left untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(ValueError):
    """Raised when a diff is malformed or does not apply to the target."""


@dataclass
class Hunk:
    old_start: int  # 1-based, as written in the header (0 for pure insertion at top)
    lines: list[tuple[str, str]] = field(default_factory=list)  # (op, text) with op in " -+"

    @property
    def old_lines(self) -> list[str]:
        return [text for op, text in self.lines if op in " -"]

    @property
    def new_lines(self) -> list[str]:
        return [text for op, text in self.lines if op in " +"]

    @property
    def added(self) -> int:
        return sum(op == "+" for op, _ in self.lines)

    @property
    def removed(self) -> int:
        return sum(op == "-" for op, _ in self.lines)


@dataclass
class PatchResult:
    new_text: str
    hunks_applied: int
    lines_added: int
    lines_removed: int
    fuzzy: bool  # True if any hunk needed offset search or whitespace-insensitive matching


def parse_unified_diff(diff: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    current: Hunk | None = None

    for raw in diff.splitlines():
        if raw.startswith("@@"):
            match = _HUNK_HEADER.match(raw)
            if not match:
                raise PatchError(f"malformed hunk header: {raw!r}")
            current = Hunk(old_start=int(match.group(1)))
            hunks.append(current)
            continue
        if current is None:
            # Preamble: "diff --git", "index", "---", "+++" and similar lines.
            continue
        if raw.startswith(("--- ", "+++ ")) and _looks_like_file_header(raw):
            # A second file's header; this applier handles one file at a time.
            raise PatchError("diff touches more than one file; send one diff per file")
        if raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        if raw == "":
            current.lines.append((" ", ""))
            continue
        op, text = raw[0], raw[1:]
        if op not in " -+":
            raise PatchError(f"unexpected line in hunk (must start with ' ', '-' or '+'): {raw!r}")
        current.lines.append((op, text))

    if not hunks:
        raise PatchError("diff contains no hunks (expected at least one '@@ -a,b +c,d @@' header)")
    for hunk in hunks:
        # Trailing blank context lines are usually an artefact of how the diff was quoted.
        while hunk.lines and hunk.lines[-1] == (" ", ""):
            hunk.lines.pop()
        if not any(op in "-+" for op, _ in hunk.lines):
            raise PatchError("hunk makes no changes")
    return hunks


def apply_unified_diff(original: str | None, diff: str) -> PatchResult:
    """Apply ``diff`` to ``original`` text. ``None`` means the file does not exist yet."""
    hunks = parse_unified_diff(diff)

    newline = "\r\n" if original and "\r\n" in original else "\n"
    had_trailing_newline = original is None or original.endswith(("\n", "\r\n")) or original == ""
    lines = (original or "").splitlines()

    fuzzy = False
    offset = 0  # net line-count change from hunks applied so far
    for index, hunk in enumerate(hunks, start=1):
        old = hunk.old_lines
        hint = max(hunk.old_start - 1, 0) + offset

        if not old:
            position, exact = min(hint, len(lines)), True
        else:
            located = _locate(lines, old, hint)
            if located is None:
                raise PatchError(_mismatch_message(index, old, lines, hint))
            position, exact = located

        if not exact or position != hint:
            fuzzy = True
        lines[position : position + len(old)] = hunk.new_lines
        offset += len(hunk.new_lines) - len(old) + (position - hint)

    text = newline.join(lines)
    if lines and had_trailing_newline:
        text += newline
    return PatchResult(
        new_text=text,
        hunks_applied=len(hunks),
        lines_added=sum(h.added for h in hunks),
        lines_removed=sum(h.removed for h in hunks),
        fuzzy=fuzzy,
    )


def _locate(lines: list[str], needle: list[str], hint: int) -> tuple[int, bool] | None:
    """Find ``needle`` in ``lines``, preferring positions closest to ``hint``."""
    for normalize, exact in ((lambda s: s, True), (lambda s: s.rstrip(), False)):
        target = [normalize(s) for s in needle]
        normalized = [normalize(s) for s in lines]
        span = len(target)
        candidates = range(0, len(lines) - span + 1)
        for position in sorted(candidates, key=lambda p: (abs(p - hint), p)):
            if normalized[position : position + span] == target:
                return position, exact
    return None


def _mismatch_message(index: int, expected: list[str], lines: list[str], hint: int) -> str:
    preview = "\n".join(f"    {line}" for line in expected[:8])
    lo = max(hint - 2, 0)
    actual = "\n".join(f"  {n + 1:>5}| {text}" for n, text in enumerate(lines[lo : lo + 8], start=lo))
    return (
        f"hunk {index} does not apply: its context/removed lines were not found in the file.\n"
        f"  expected (context + '-' lines):\n{preview}\n"
        f"  file near line {hint + 1}:\n{actual}\n"
        "Re-read the file and regenerate the diff from its current contents."
    )


def _looks_like_file_header(line: str) -> bool:
    # "--- a/foo.c" / "+++ b/foo.c" - distinguishes a header from a removed line starting "-- ".
    return bool(re.match(r"^(---|\+\+\+) (a/|b/|/dev/null|\S+\.\w+)", line))
