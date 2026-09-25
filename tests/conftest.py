import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from autofix.config import ProjectConfig

HAS_CC = shutil.which("gcc") is not None or shutil.which("cc") is not None


def pytest_collection_modifyitems(config, items):
    if HAS_CC:
        return
    skip = pytest.mark.skip(reason="no C compiler on PATH")
    for item in items:
        if "needs_cc" in item.keywords:
            item.add_marker(skip)


def make_scripted_project(root: Path, *, build_script: str, test_script: str, protected=("tests/**",)) -> ProjectConfig:
    """A project whose 'build' and 'test' steps are small Python scripts.

    Lets tool and agent tests exercise real subprocesses without a C toolchain.
    """
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "tools" / "build.py").write_text(textwrap.dedent(build_script), encoding="utf-8")
    (root / "tools" / "test.py").write_text(textwrap.dedent(test_script), encoding="utf-8")
    data = {
        "build": {"command": [sys.executable, "tools/build.py"], "timeout": 30},
        "test": {"command": [sys.executable, "tools/test.py"], "timeout": 30},
        "sandbox": {"protected": list(protected) + ["tools/**"]},
    }
    return ProjectConfig.from_dict(root, data)


# A toy "compiler": fails if src/calc.c still contains the seeded bug marker.
TOY_BUILD = """
    import pathlib, sys
    src = pathlib.Path("src/calc.c").read_text()
    if "retrun" in src:
        print("src/calc.c:3:5: error: unknown type name 'retrun'")
        sys.exit(1)
    print("compiled 1 file")
"""

# A toy test runner: passes iff calc.c adds rather than subtracts.
TOY_TEST = """
    import pathlib, sys
    args = sys.argv[1:]
    src = pathlib.Path("src/calc.c").read_text()
    tests = {"add_small": "a + b" in src, "add_zero": True}
    failed = 0
    for name, ok in tests.items():
        if args and args[0] not in name:
            continue
        print(f"RUN  {name}")
        if ok:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name} (tests/test_calc.c:7): expected 5, got -1")
    print(f"SUMMARY passed={len(tests) - failed} failed={failed}")
    sys.exit(1 if failed else 0)
"""


@pytest.fixture
def toy_project(tmp_path):
    root = tmp_path / "calc"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "calc.c").write_text(
        "#include \"calc.h\"\n\nint add(int a, int b) {\n    return a - b;\n}\n", encoding="utf-8"
    )
    (root / "src" / "calc.h").write_text("int add(int a, int b);\n", encoding="utf-8")
    (root / "tests" / "test_calc.c").write_text("/* tests */\n", encoding="utf-8")
    return make_scripted_project(root, build_script=TOY_BUILD, test_script=TOY_TEST)
