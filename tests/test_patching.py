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
        ("@@ -x,1 +1 @@\n-a\n+b\n", "anchor line not found"),
        ("@@ -1,1 +1,1 @@\n a\n", "no changes"),
        ("@@ -1,1 +1,1 @@\n*a\n", "unexpected line"),
        ("@@ -1 +1 @@\n-a\n+b\n--- a/other.c\n+++ b/other.c\n@@ -1 +1 @@\n-x\n+y\n", "more than one file"),
    ],
)
def test_rejects_malformed_diffs(diff, message):
    with pytest.raises(PatchError, match=message):
        apply_unified_diff("x\n", diff)


# -- context-anchored patch format -------------------------------------------------

from autofix.patching import patch_target  # noqa: E402


def test_anchored_format_without_line_numbers():
    diff = """\
*** Begin Patch
*** Update File: src/ring.c
@@
-    if (r->count > r->cap) {
+    if (r->count >= r->cap) {
*** End Patch"""
    result = apply_unified_diff(SOURCE, diff)
    assert "if (r->count >= r->cap) {" in result.new_text
    assert result.hunks_applied == 1


def test_anchored_format_uses_anchor_line_and_order():
    source = "int a(void) {\n    return 0;\n}\n\nint b(void) {\n    return 0;\n}\n"
    diff = """\
*** Begin Patch
*** Update File: x.c
@@ int b(void) {
-    return 0;
+    return 2;
*** End Patch"""
    assert apply_unified_diff(source, diff).new_text.splitlines()[5] == "    return 2;"

    two = """\
*** Update File: x.c
@@
-    return 0;
+    return 1;
@@
-    return 0;
+    return 2;
"""
    lines = apply_unified_diff(source, two).new_text.splitlines()
    assert lines[1] == "    return 1;" and lines[5] == "    return 2;"


def test_anchored_format_changes_directly_after_directive():
    diff = "*** Begin Patch\n*** Update File: ring.c\n-    return 0;\n+    return 1;\n*** End Patch\n"
    assert "    return 1;" in apply_unified_diff(SOURCE, diff).new_text


def test_anchored_format_add_file():
    diff = "*** Begin Patch\n*** Add File: new.h\n+#pragma once\n+int f(void);\n*** End Patch\n"
    assert apply_unified_diff(None, diff).new_text == "#pragma once\nint f(void);\n"


@pytest.mark.parametrize(
    "diff, message",
    [
        ("*** Begin Patch\n*** Delete File: a.c\n*** End Patch", "deleting files"),
        ("*** Update File: a.c\n@@\n-x\n+y\n*** Update File: b.c\n@@\n-x\n+y\n", "more than one file"),
        ("*** Move to: b.c\n", "unsupported patch directive"),
        ("*** Update File: a.c\n@@ no_such_line\n-x\n+y\n", "anchor line not found"),
    ],
)
def test_anchored_format_errors(diff, message):
    with pytest.raises(PatchError, match=message):
        apply_unified_diff("x\n", diff)


@pytest.mark.parametrize(
    "diff, expected",
    [
        ("*** Begin Patch\n*** Update File: src/ring.c\n@@\n-a\n+b\n", "src/ring.c"),
        ("--- a/src/ring.c\n+++ b/src/ring.c\n@@ -1 +1 @@\n-a\n+b\n", "src/ring.c"),
        ("--- /dev/null\n+++ b/new.h\n@@ -0,0 +1 @@\n+a\n", "new.h"),
        ("@@ -1 +1 @@\n-a\n+b\n", None),
    ],
)
def test_patch_target(diff, expected):
    assert patch_target(diff) == expected


# -- quirks seen in real model output ---------------------------------------------

LRU = """\
bool Cache::contains(int key) const
{
    return index_[key] != order_.end();
}
"""


def test_bare_hunk_header_in_plain_diff():
    diff = "--- a/src/ring.c\n+++ b/src/ring.c\n@@\n-    return 0;\n+    return 1;\n"
    assert "    return 1;" in apply_unified_diff(SOURCE, diff).new_text


def test_line_number_prefixes_from_listings_are_stripped():
    source = "int ring_size(const ring_t *r)\n{\n    return r->count;\n}\n"
    diff = (
        "--- a/src/ring.c\n@@\n-67| int ring_size(const ring_t *r)\n-68| {\n-69|     return r->count;\n"
        "-70| }\n+67| size_t ring_size(const ring_t *r)\n+68| {\n+69|     return r->count;\n+70| }\n"
    )
    assert apply_unified_diff(source, diff).new_text == source.replace("int ring_size", "size_t ring_size")


def test_numbered_looking_code_is_not_stripped_unless_every_line_is():
    source = "x = 1| 2;\ny = 3;\n"
    diff = "@@\n-x = 1| 2;\n+x = 1| 4;\n y = 3;\n"
    assert apply_unified_diff(source, diff).new_text == "x = 1| 4;\ny = 3;\n"


def test_uniformly_over_indented_hunk_is_reindented():
    diff = (
        "@@ -38,5 +38,7 @@\n-    bool Cache::contains(int key) const\n-    {\n"
        "-        return index_[key] != order_.end();\n-    }\n+    bool Cache::contains(int key) const\n"
        "+    {\n+        auto it = index_.find(key);\n+        return it != index_.end();\n+    }\n"
    )
    result = apply_unified_diff(LRU, diff)
    assert result.new_text == (
        "bool Cache::contains(int key) const\n{\n    auto it = index_.find(key);\n"
        "    return it != index_.end();\n}\n"
    )
    assert result.fuzzy


def test_under_indented_hunk_is_reindented():
    source = "void f(void)\n{\n    if (x) {\n        y();\n    }\n}\n"
    diff = "@@\n-if (x) {\n-    y();\n-}\n+if (x && z) {\n+    y();\n+}\n"
    assert apply_unified_diff(source, diff).new_text == (
        "void f(void)\n{\n    if (x && z) {\n        y();\n    }\n}\n"
    )


def test_exact_match_preferred_over_indentation_match():
    source = "    a();\na();\n"
    diff = "@@\n-a();\n+b();\n"
    assert apply_unified_diff(source, diff).new_text == "    a();\nb();\n"
