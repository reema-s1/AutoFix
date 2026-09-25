"""The redaction guarantees: nothing credential-shaped, no secret env value and
no path outside the project may survive, while ordinary compiler output and C
source must come through intact."""

import os

import pytest

from autofix.redaction import Redactor


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    return root


@pytest.fixture
def redactor(project):
    env = {
        "HOME": "/home/alice",
        "GITHUB_TOKEN": "tok_abcdef1234567890",
        "DB_PASSWORD": "correct-horse-battery",
        "SHORT_KEY": "abc",  # too short to redact safely
        "PATH": "/usr/bin:/bin",
    }
    return Redactor(project_root=project, env=env)


SECRETS = [
    "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    "sk-proj-AbCdEfGhIjKlMnOpQrStUvWx",
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_" + "a" * 36,
    "github_pat_" + "B" * 50,
    "xoxb-1234567890-abcdefghij",
    "AIza" + "x" * 35,
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
]


@pytest.mark.parametrize("secret", SECRETS)
def test_credential_shapes_are_redacted(redactor, secret):
    out = redactor.redact(f"config loaded: {secret} (ok)")
    assert secret not in out
    assert "<redacted:" in out
    assert out.startswith("config loaded: ") and out.endswith(" (ok)")


def test_bearer_token_keeps_scheme(redactor):
    out = redactor.redact("Authorization: Bearer abcdefghijklmnop.qrstuvwx")
    assert out == "Authorization: Bearer <redacted:bearer-token>"


def test_url_credentials(redactor):
    out = redactor.redact("fetching https://deploy:s3cr3tpass@git.example.com/repo.git")
    assert "s3cr3tpass" not in out
    assert "https://deploy:<redacted:url-credentials>@git.example.com" in out


def test_quoted_secret_assignment(redactor):
    out = redactor.redact('static const char *db_password = "hunter2hunter2";')
    assert "hunter2hunter2" not in out
    assert out.startswith("static const char *db_password = \"<redacted:")


def test_env_style_assignment(redactor):
    out = redactor.redact("export STRIPE_API_KEY=rk_live_0123456789")
    assert out == "export STRIPE_API_KEY=<redacted:secret-assignment>"


def test_private_key_block(redactor):
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nIBAAK\n-----END RSA PRIVATE KEY-----"
    assert redactor.redact(f"key:\n{pem}\ndone") == "key:\n<redacted:private-key>\ndone"


def test_sensitive_env_values_redacted_by_name(redactor):
    out = redactor.redact("clone with tok_abcdef1234567890 and correct-horse-battery")
    assert out == "clone with <redacted:env:GITHUB_TOKEN> and <redacted:env:DB_PASSWORD>"


def test_short_or_non_sensitive_env_values_untouched(redactor):
    assert redactor.redact("value abc here") == "value abc here"


def test_project_root_becomes_relative(redactor, project):
    for root in {str(project), project.as_posix()}:
        line = f"{root}{os.sep}src{os.sep}ring.c:12:5: error: expected ';'"
        out = redactor.redact(line)
        assert out.replace("\\", "/") == "src/ring.c:12:5: error: expected ';'"
    assert redactor.redact(f"cd {project.as_posix()} && make") == "cd . && make"


def test_external_paths_reduced_to_basename(redactor):
    out = redactor.redact("In file included from /usr/include/stdio.h:27:")
    assert out == "In file included from <external>/stdio.h:27:"
    out = redactor.redact(r"note: C:\Users\alice\secret-project\notes.txt:1")
    assert "alice" not in out and out.endswith("<external>/notes.txt:1")
    assert "alice" not in redactor.redact("home is /home/alice/")


@pytest.mark.parametrize(
    "benign",
    [
        "src/ring.c:12:5: error: 'count' undeclared (first use in this function)",
        "    token = next_token(p);",
        "int key = hash(buf) % 64;",
        "ratio = 3/4; x = a / b;",
        "see https://example.com/docs/page for details",
        "test_ring_push ............. FAIL",
        "char *password_prompt = read_line();",
    ],
)
def test_benign_text_is_untouched(redactor, benign):
    assert redactor.redact(benign) == benign


def test_redact_obj_recurses(redactor):
    event = {"args": {"path": "src/a.c"}, "out": ["AKIAIOSFODNN7EXAMPLE", 3], "n": 1}
    out = redactor.redact_obj(event)
    assert out == {"args": {"path": "src/a.c"}, "out": ["<redacted:aws-access-key>", 3], "n": 1}


def test_extra_secrets(project):
    r = Redactor(project_root=project, env={}, extra_secrets=("pa55phrase-for-ci",))
    assert r.redact("using pa55phrase-for-ci") == "using <redacted:secret>"
