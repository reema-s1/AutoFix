import pytest

from autofix.patching import PatchError, apply_unified_diff, parse_unified_diff

SOURCE = """\
#include "ring.h"

int ring_push(ring_t *r, int v) {
    if (r->count > r->cap) {
        return -1;
    }
    r->buf[r->head] = v;
    r->head = (r->head + 1) % r->cap;
    r->count++;
    return 0;
}
"""


def test_applies_exact_hunk():
    diff = """\
--- a/ring.c
+++ b/ring.c
@@ -4,3 +4,3 @@
     if (r->count > r->cap) {
-        return -1;
+        return -2;
     }
"""
    result = apply_unified_diff(SOURCE, diff)
    assert "return -2;" in result.new_text
    assert "return -1;" not in result.new_text
    assert result.lines_added == 1 and result.lines_removed == 1
    assert not result.fuzzy


def test_wrong_line_numbers_are_tolerated():
    diff = """\
@@ -40,1 +40,1 @@
-    if (r->count > r->cap) {
+    if (r->count >= r->cap) {
"""
    result = apply_unified_diff(SOURCE, diff)
    assert "if (r->count >= r->cap) {" in result.new_text
    assert result.fuzzy


def test_wrong_hunk_counts_are_ignored():
    diff = """\
@@ -4,99 +4,1 @@
-    if (r->count > r->cap) {
+    if (r->count >= r->cap) {
"""
    assert "r->count >= r->cap" in apply_unified_diff(SOURCE, diff).new_text


def test_trailing_whitespace_insensitive_fallback():
    diff = "@@ -10,1 +10,1 @@\n-    return 0;   \n+    return 1;\n"
    result = apply_unified_diff(SOURCE, diff)
    assert "    return 1;" in result.new_text
    assert result.fuzzy


def test_multiple_hunks_track_offsets():
    diff = """\
@@ -1,1 +1,2 @@
 #include "ring.h"
+#include <stddef.h>
@@ -9,2 +10,2 @@
     r->count++;
-    return 0;
+    return 1;
"""
    result = apply_unified_diff(SOURCE, diff)
    lines = result.new_text.splitlines()
    assert lines[1] == "#include <stddef.h>"
    assert lines[10] == "    return 1;"
    assert result.hunks_applied == 2


def test_failed_hunk_leaves_nothing_applied():
    diff = """\
@@ -1,1 +1,1 @@
-#include "ring.h"
+#include "ring2.h"
@@ -5,1 +5,1 @@
-        return -99;
+        return -1;
"""
    with pytest.raises(PatchError, match="hunk 2 does not apply"):
        apply_unified_diff(SOURCE, diff)


def test_blank_lines_inside_hunk_are_context():
    diff = '@@ -1,3 +1,3 @@\n #include "ring.h"\n\n-int ring_push(ring_t *r, int v) {\n+int ring_push(ring_t *r, int value) {\n'
    assert "int value" in apply_unified_diff(SOURCE, diff).new_text


def test_preserves_crlf_and_missing_trailing_newline():
    original = "a\r\nb\r\nc"
    result = apply_unified_diff(original, "@@ -2,1 +2,1 @@\n-b\n+B\n")
    assert result.new_text == "a\r\nB\r\nc"


def test_creates_new_file():
    diff = "--- /dev/null\n+++ b/new.h\n@@ -0,0 +1,2 @@\n+#pragma once\n+int f(void);\n"
    assert apply_unified_diff(None, diff).new_text == "#pragma once\nint f(void);\n"


@pytest.mark.parametrize(
    "diff, message",
    [
        ("just some text", "no hunks"),
        ("@@ bogus @@\n-a\n", "malformed hunk header"),
        ("@@ -1,1 +1,1 @@\n a\n", "no changes"),
        ("@@ -1,1 +1,1 @@\n*a\n", "unexpected line"),
        ("@@ -1 +1 @@\n-a\n+b\n--- a/other.c\n+++ b/other.c\n@@ -1 +1 @@\n-x\n+y\n", "more than one file"),
    ],
)
def test_rejects_malformed_diffs(diff, message):
    with pytest.raises(PatchError, match=message):
        parse_unified_diff(diff)
