from pathlib import Path

import pytest

from autofix.config import ConfigError, ProjectConfig


def write_config(root: Path, text: str) -> None:
    (root / "autofix.toml").write_text(text, encoding="utf-8")


def test_load_minimal_config(tmp_path):
    write_config(
        tmp_path,
        """
[build]
command = ["gcc", "-o", "build/t", "main.c"]

[test]
command = ["build/t"]
""",
    )
    cfg = ProjectConfig.load(tmp_path)
    assert cfg.root == tmp_path.resolve()
    assert cfg.build.argv == ("gcc", "-o", "build/t", "main.c")
    assert cfg.build.timeout == 300
    assert cfg.test_format == "minitest"
    assert cfg.protected == ()


def test_load_full_config(tmp_path):
    write_config(
        tmp_path,
        """
[build]
command = ["cmake", "--build", "build"]
timeout = 30

[test]
command = ["ctest"]
format = "exit-code"
timeout = 10

[sandbox]
protected = ["tests/**"]
source_globs = ["**/*.c"]
""",
    )
    cfg = ProjectConfig.load(tmp_path)
    assert cfg.build.timeout == 30
    assert cfg.test_format == "exit-code"
    assert cfg.protected == ("tests/**",)
    assert cfg.source_globs == ("**/*.c",)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="no autofix.toml"):
        ProjectConfig.load(tmp_path)


@pytest.mark.parametrize(
    "text, message",
    [
        ("[test]\ncommand=['x']\n", r"\[build\]"),
        ("[build]\ncommand='gcc'\n[test]\ncommand=['x']\n", "list of strings"),
        ("[build]\ncommand=[]\n[test]\ncommand=['x']\n", "non-empty"),
        ("[build]\ncommand=['a']\n[test]\ncommand=['x']\nformat='junit'\n", "test.format"),
        ("[build\n", "autofix.toml"),
    ],
)
def test_invalid_configs(tmp_path, text, message):
    write_config(tmp_path, text)
    with pytest.raises(ConfigError, match=message):
        ProjectConfig.load(tmp_path)
