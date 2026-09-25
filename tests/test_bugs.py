from pathlib import Path

import pytest

from autofix.bugs import BugSpecError, load_bug, load_bugs, materialize
from autofix.config import ProjectConfig
from autofix.tools import Toolbox

ROOT = Path(__file__).resolve().parent.parent
BUGS_DIR = ROOT / "fixtures" / "bugs"
FIXTURES = ROOT / "fixtures"
ALL_BUGS = load_bugs(BUGS_DIR)


def test_catalog_shape():
    assert len(ALL_BUGS) == 16
    assert {b.project for b in ALL_BUGS} == {"ringbuf", "slidewin", "strkit", "lrucache"}
    assert {b.category for b in ALL_BUGS} == {"build", "logic", "crash", "memory"}
    assert len({b.id for b in ALL_BUGS}) == len(ALL_BUGS)
    for bug in ALL_BUGS:
        assert bug.keywords, bug.id
        assert bug.keyword_match(bug.root_cause), f"{bug.id}: keywords must match its own root cause"


def test_filter_by_id_or_project():
    assert [b.id for b in load_bugs(BUGS_DIR, {"ringbuf-peek-index"})] == ["ringbuf-peek-index"]
    assert {b.project for b in load_bugs(BUGS_DIR, {"strkit"})} == {"strkit"}
    with pytest.raises(BugSpecError, match="unknown"):
        load_bugs(BUGS_DIR, {"nope"})


def test_keyword_match_requires_every_group():
    bug = next(b for b in ALL_BUGS if b.id == "ringbuf-full-check")
    assert bug.keyword_match("The capacity check in ring_push uses > instead of ==")
    assert not bug.keyword_match("ring_push is broken")
    assert not bug.keyword_match("the ring is full")


def test_invalid_bug_specs(tmp_path):
    bad = tmp_path / "x.toml"
    bad.write_text('id = "y"\nproject = "p"\ncategory = "logic"\nsymptom = "crash"\nroot_cause = "r"\n'
                   '[[mutation]]\nfile = "a.c"\nfind = "a"\nreplace = "b"\n')
    with pytest.raises(BugSpecError, match="must match the file name"):
        load_bug(bad)
    bad.write_text('id = "x"\nproject = "p"\ncategory = "typo"\nsymptom = "crash"\nroot_cause = "r"\n'
                   '[[mutation]]\nfile = "a.c"\nfind = "a"\nreplace = "b"\n')
    with pytest.raises(BugSpecError, match="category"):
        load_bug(bad)


def test_materialize_requires_unique_match(tmp_path):
    bug = next(b for b in ALL_BUGS if b.id == "ringbuf-peek-index")
    fixtures = tmp_path / "fixtures"
    project = fixtures / "ringbuf"
    project.mkdir(parents=True)
    (project / "autofix.toml").write_text("")
    (project / "src").mkdir()
    find = bug.mutations[0].find
    (project / "src" / "ring.c").write_text(find + "\n" + find)
    with pytest.raises(BugSpecError, match="exactly once"):
        materialize(bug, fixtures, tmp_path / "work")


def test_materialize_leaves_fixture_untouched(tmp_path):
    bug = ALL_BUGS[0]
    source = FIXTURES / bug.project / bug.mutations[0].file
    before = source.read_bytes()
    work = materialize(bug, FIXTURES, tmp_path / "work")
    assert source.read_bytes() == before
    assert bug.mutations[0].replace in (work / bug.mutations[0].file).read_text()


@pytest.mark.needs_cc
@pytest.mark.parametrize("bug", ALL_BUGS, ids=lambda b: b.id)
def test_each_bug_produces_its_declared_symptom(bug, tmp_path):
    work = materialize(bug, FIXTURES, tmp_path / bug.id)
    toolbox = Toolbox(ProjectConfig.load(work))
    report = toolbox.test_report()
    if bug.symptom == "build_failure":
        assert report is None, "expected the build to fail"
        return
    assert report is not None, "expected the build to succeed"
    assert not report.all_passed, "expected at least one failing test"
    if bug.symptom == "crash":
        assert any(c.status == "crash" for c in report.failed), report.summary_line()


# Patches written by models during real evaluation runs that the applier once
# rejected. Each must now repair its bug so that the full test suite passes.
RECORDED_MODEL_PATCHES = [
    (
        "lru-const-contains",
        "src/lru.cpp",
        "@@ -38,5 +38,7 @@\n-    bool Cache::contains(int key) const\n-    {\n"
        "-        return index_[key] != order_.end();\n-    }\n+    bool Cache::contains(int key) const\n"
        "+    {\n+        // Use find to avoid modifying the map in a const context.\n"
        "+        auto it = index_.find(key);\n+        return it != index_.end();\n+    }\n",
    ),
    (
        "ringbuf-size-signature",
        "src/ring.c",
        "--- a/src/ring.c\n@@\n-67| int ring_size(const ring_t *r)\n-68| {\n-69|     return r->count;\n"
        "-70| }\n+67| size_t ring_size(const ring_t *r)\n+68| {\n+69|     return r->count;\n+70| }\n",
    ),
    (
        "ringbuf-size-signature",
        "src/ring.c",
        "--- a/src/ring.c\n+++ b/src/ring.c\n@@\n-    int ring_size(const ring_t *r)\n-    {\n"
        "-        return r->count;\n-    }\n+    size_t ring_size(const ring_t *r)\n+    {\n"
        "+        return r->count;\n+    }\n",
    ),
    (
        "ringbuf-full-check",
        None,
        "*** Begin Patch\n*** Update File: src/ring.c\n@@\n-    if (r->count > r->cap) {\n"
        "-        return -1;\n-    }\n+    if (r->count >= r->cap) {\n+        return -1;\n+    }\n"
        "*** End Patch",
    ),
]


@pytest.mark.needs_cc
@pytest.mark.parametrize("bug_id, path, diff", RECORDED_MODEL_PATCHES, ids=lambda v: str(v)[:24])
def test_recorded_model_patches_repair_their_bug(bug_id, path, diff, tmp_path):
    bug = next(b for b in ALL_BUGS if b.id == bug_id)
    toolbox = Toolbox(ProjectConfig.load(materialize(bug, FIXTURES, tmp_path / bug_id)))
    args = {"diff": diff} if path is None else {"path": path, "diff": diff}
    result = toolbox.call("apply_patch", args)
    assert not result.is_error, result.content
    report = toolbox.test_report()
    assert report is not None and report.all_passed
