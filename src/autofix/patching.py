"""A forgiving diff applier for model-authored patches.

Language models are good at writing the *content* of a hunk but unreliable
at line numbers and hunk-header counts. This applier therefore:

* ignores the ``+c,d`` counts and derives hunk extent from line prefixes,
* treats the ``-a`` start line only as a hint, searching outward from it for
  the hunk's context + removed lines,
* falls back to a trailing-whitespace-insensitive match,
* applies all hunks atomically: either every hunk lands or the file is
  left untouched.

Two input formats are accepted, since models are trained on both:

* standard unified diffs (``--- a/f``, ``+++ b/f``, ``@@ -a,b +c,d @@``), and
* the context-anchored patch format some models emit natively::

      *** Begin Patch
      *** Update File: src/ring.c
      @@ int ring_push(ring_t *r, int value)
      -    if (r->count > r->cap) {
      +    if (r->count == r->cap) {
      *** End Patch

  Here hunks carry no line numbers; an optional ``@@ <text>`` header names a
  line the change follows, and hunks are applied in file order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_DIRECTIVE = re.compile(r"^\*\*\* (Update|Add|Delete) File:\s*(.+?)\s*$")
_IGNORED_DIRECTIVES = ("*** Begin Patch", "*** End Patch", "*** End of File")


class PatchError(ValueError):
    """Raised when a diff is malformed or does not apply to the target."""


@dataclass
class Hunk:
    # 1-based start line from a unified-diff header (0 for insertion at the top);
    # None when the format carries no line numbers.
    old_start: int | None
    anchor: str | None = None  # text of a line the hunk follows ("@@ <anchor>")
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


def patch_target(diff: str) -> str | None:
    """The file a diff names in its headers, if any."""
    for raw in diff.splitlines():
        if match := _FILE_DIRECTIVE.match(raw):
            return match.group(2)
    for prefix in ("+++ ", "--- "):
        for raw in diff.splitlines():
            if raw.startswith(prefix) and _looks_like_file_header(raw):
                name = raw[4:].split("\t")[0].strip()
                if name != "/dev/null":
                    return name[2:] if name.startswith(("a/", "b/")) else name
    return None


def parse_unified_diff(diff: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    current: Hunk | None = None
    anchored_file: str | None = None  # set once an "*** Update/Add File:" directive is seen

    for raw in diff.splitlines():
        if raw.startswith("*** "):
            if raw.strip() in _IGNORED_DIRECTIVES:
                continue
            match = _FILE_DIRECTIVE.match(raw)
            if match is None:
                raise PatchError(f"unsupported patch directive: {raw!r}")
            if match.group(1) == "Delete":
                raise PatchError("deleting files is not supported")
            if anchored_file is not None and anchored_file != match.group(2):
                raise PatchError("diff touches more than one file; send one diff per file")
            anchored_file = match.group(2)
            current = None
            continue
        if raw.startswith("@@"):
            match = _HUNK_HEADER.match(raw)
            if match:
                current = Hunk(old_start=int(match.group(1)))
            elif anchored_file is not None:
                current = Hunk(old_start=None, anchor=raw[2:].strip(" @") or None)
            else:
                raise PatchError(f"malformed hunk header: {raw!r} (expected '@@ -a,b +c,d @@')")
            hunks.append(current)
            continue
        if current is None:
            if anchored_file is not None and raw[:1] in (" ", "-", "+"):
                # Anchored format allows changes directly after the file directive.
                current = Hunk(old_start=None)
                hunks.append(current)
            else:
                # Preamble: "diff --git", "index", "---", "+++" and similar lines.
                continue
        if raw.startswith(("--- ", "+++ ")) and _looks_like_file_header(raw):
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
    offset = 0  # net line-count drift between header line numbers and the file
    cursor = 0  # end of the previously applied hunk, for hunks without line numbers
    for index, hunk in enumerate(hunks, start=1):
        old = hunk.old_lines
        if hunk.old_start is not None:
            hint = max(hunk.old_start - 1, 0) + offset
            if not old:
                position, exact = min(hint, len(lines)), True
            else:
                located = _locate_near(lines, old, hint)
                if located is None:
                    raise PatchError(_mismatch_message(index, old, lines, hint))
                position, exact = located
            if not exact or position != hint:
                fuzzy = True
            offset += len(hunk.new_lines) - len(old) + (position - hint)
        else:
            start = cursor
            if hunk.anchor:
                anchor_at = _find_anchor(lines, hunk.anchor, cursor)
                if anchor_at is None:
                    raise PatchError(f"hunk {index}: anchor line not found: {hunk.anchor!r}")
                start = anchor_at + 1
            if not old:
                position, exact = (start if hunk.anchor else len(lines)), True
            else:
                located = _locate_after(lines, old, start)
                if located is None:
                    raise PatchError(_mismatch_message(index, old, lines, start))
                position, exact = located
            fuzzy = fuzzy or not exact

        lines[position : position + len(old)] = hunk.new_lines
        cursor = position + len(hunk.new_lines)

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


def _matches(lines: list[str], needle: list[str]) -> tuple[list[int], bool]:
    """All positions of ``needle`` in ``lines``: exact matches, else whitespace-insensitive."""
    span = len(needle)
    for normalize, exact in ((lambda s: s, True), (lambda s: s.rstrip(), False)):
        target = [normalize(s) for s in needle]
        normalized = [normalize(s) for s in lines]
        positions = [p for p in range(len(lines) - span + 1) if normalized[p : p + span] == target]
        if positions:
            return positions, exact
    return [], True


def _locate_near(lines: list[str], needle: list[str], hint: int) -> tuple[int, bool] | None:
    """The match closest to ``hint``."""
    positions, exact = _matches(lines, needle)
    if not positions:
        return None
    return min(positions, key=lambda p: (abs(p - hint), p)), exact


def _locate_after(lines: list[str], needle: list[str], start: int) -> tuple[int, bool] | None:
    """The first match at or after ``start``, else the first match anywhere."""
    positions, exact = _matches(lines, needle)
    if not positions:
        return None
    after = [p for p in positions if p >= start]
    return (after or positions)[0], exact


def _find_anchor(lines: list[str], anchor: str, start: int) -> int | None:
    wanted = anchor.strip()
    for pool in (range(start, len(lines)), range(0, len(lines))):
        for n in pool:
            if lines[n].strip() == wanted or (wanted and wanted in lines[n]):
                return n
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
