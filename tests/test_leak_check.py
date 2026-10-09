"""Tests for tools/leak_check.py, which scans tracked files' working-tree bytes for private values."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dgxm_tools_leak_check", REPO / "tools" / "leak_check.py"
)
assert SPEC and SPEC.loader
leak_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = leak_check
SPEC.loader.exec_module(leak_check)


def _ascii(*codes: int) -> bytes:
    return bytes(codes)


def _empty_allowlist() -> str:
    return "version = 1\nallow = []\n"


def _init_repo(
    root: Path,
    files: dict[str, str | bytes],
    *,
    allowlist: str | None = None,
) -> Path:
    subprocess.run(
        ["git", "init", "-q"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    allowlist_path = root / "tools" / "leak_allowlist.toml"
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    allowlist_path.write_text(allowlist or _empty_allowlist(), encoding="utf-8")
    subprocess.run(
        ["git", "add", "-A"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return allowlist_path


def _finding_keys(result) -> set[tuple[str, int, str]]:
    return {(item.path, item.line, item.rule) for item in result.findings}


def test_structural_workstation_residue_rules_need_no_private_literals(tmp_path: Path):
    rig_hostname = b"edge" + b"compute-a1b2"
    private_home = b"/home/" + b"project-owner"
    custom_key = b"id_ed25519_" + b"cluster-admin"
    allowlist = _init_repo(
        tmp_path,
        {
            "notes.txt": b"host=" + rig_hostname + b" home=" + private_home + b"\n",
            os.fsdecode(custom_key): b"fixture\n",
        },
    )

    result = leak_check.scan_repository(tmp_path, allowlist)
    keys = _finding_keys(result)

    assert keys == {
        (os.fsdecode(custom_key), 1, "key-filename"),
        ("notes.txt", 1, "private-home"),
        ("notes.txt", 1, "rig-hostname"),
    }


@pytest.mark.parametrize(
    "path",
    (
        b"/home/user/ComfyUI",
        b"/home/u/ComfyUI",
        b"/home/alice/ComfyUI",
        b"/home/operator/ComfyUI",
        b"/home/worker/ComfyUI",
        b"/home/example/ComfyUI",
        b"/home/runner/ComfyUI",
        b"/Users/user/ComfyUI",
        b"C:\\Users\\user\\ComfyUI",
        b"c:/users/user/ComfyUI",
        b"\\\\example-host\\Users\\operator\\ComfyUI",
        b"/home/operator?credential=redacted",
        b'quoted=/home/alice"',
    ),
)
def test_documented_placeholder_home_paths_are_not_findings(path: bytes):
    assert "private-home" not in {
        item.rule for item in leak_check.scan_bytes("fixture.txt", path)
    }


@pytest.mark.parametrize(
    "path",
    (
        b"D:/Users/" + b"project-owner" + b"/ComfyUI",
        b"\\\\build-host\\Users\\" + b"project-owner" + b"\\ComfyUI",
    ),
)
def test_non_placeholder_windows_home_paths_are_findings(path: bytes):
    assert "private-home" in {
        item.rule for item in leak_check.scan_bytes("fixture.txt", path)
    }


def test_plain_default_ssh_key_filename_is_not_a_finding():
    default_key = b"~/.ssh/" + b"id_ed25519"

    assert "key-filename" not in {
        item.rule for item in leak_check.scan_bytes("fixture.txt", default_key)
    }


@pytest.mark.parametrize(
    "address",
    (
        b"user@example.com",
        b"user@subdomain.example.net",
        b"user@example.org",
        b"user@docs.invalid",
        b"user@docs.test",
        b"user@docs.example",
        b"USER@EXAMPLE.COM",
    ),
)
def test_reserved_example_email_domains_are_not_findings(address: bytes):
    assert "email-address" not in {
        item.rule for item in leak_check.scan_bytes("fixture.txt", address)
    }


def test_scanner_source_uses_structural_rules_not_encoded_private_constants():
    source = (REPO / "tools" / "leak_check.py").read_text()

    for obsolete in (
        "_PRIVATE_HOST_PREFIX",
        "_PRIVATE_HOST_IDS",
        "_PRIVATE_HOME =",
        "\n_PRIVATE_KEY_FILENAME =",
        "known-private values",
    ):
        assert obsolete not in source


def test_unrelated_short_hex_values_are_not_machine_identity_findings():
    data = b"build a1b2 palette c3d4f5 revision deadbeef"

    assert not {item.rule for item in leak_check.scan_bytes("fixture.txt", data)} & {
        "rig-hostname",
        "short-host-id",
    }


def test_custom_private_key_filename_is_caught_in_a_path(tmp_path: Path):
    private_filename = b"id_ed25519_" + b"deployment"
    relative = os.fsdecode(private_filename)
    allowlist = _init_repo(tmp_path, {relative: "fixture\n"})

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert (relative, 1, "key-filename") in _finding_keys(result)


def test_rig_style_hostname_requires_a_machine_prefix_and_hex_suffix():
    hostname = b"spark" + b"station-fa1e"
    findings = leak_check.scan_bytes("fixture.txt", hostname)

    assert {item.rule for item in findings} == {"rig-hostname"}
    assert not any(
        item.rule == "rig-hostname"
        for item in leak_check.scan_bytes(
            "fixture.txt",
            b"spark-a spark-1 edge-case spark-interface spark-cafe dgx-monarch",
        )
    )


def _generic_samples() -> tuple[tuple[str, bytes], ...]:
    huggingface_prefix = _ascii(104, 102, 95)
    fine_grained_github_prefix = _ascii(
        103, 105, 116, 104, 117, 98, 95, 112, 97, 116, 95
    )
    aws_access_key_prefix = _ascii(65, 75, 73, 65)
    return (
        ("private-home", b"path=" + b"/home/" + b"project-owner" + b"/x"),
        ("network-address", b"peer=100." + b"89.12.34"),
        ("email-address", b"owner=" + b"person" + b"@" + b"company.local"),
        ("github-token", b"token=" + b"ghp_" + b"A" * 36),
        ("github-token", b"joined" + b"ghp_" + b"B" * 36),
        (
            "github-fine-grained-token",
            b"token=" + fine_grained_github_prefix + b"A" * 22 + b"_" + b"B" * 59,
        ),
        ("huggingface-token", b"token=" + huggingface_prefix + b"C" * 34),
        ("aws-access-key-id", b"key=" + aws_access_key_prefix + b"D" * 16),
        ("api-key", b"token=" + b"sk-" + b"a1" * 16),
        ("api-key", b"joined" + b"sk-" + b"b2" * 16),
        ("slack-token", b"token=" + b"xoxb-" + b"1234567890-abcdef"),
        ("slack-token", b"token=" + b"xoxc-" + b"1234567890-abcdef"),
        ("slack-token", b"joined" + b"xoxc-" + b"1234567890-abcdef"),
        ("private-key", b"-----BEGIN " + b"PRIVATE KEY-----"),
        (
            "gpu-identifier",
            b"GPU-" + b"01234567-89ab-cdef-0123-456789abcdef",
        ),
        ("gpu-identifier", b"GPU-" + b"01234567-redacted"),
    )


@pytest.mark.parametrize(("rule", "sample"), _generic_samples())
def test_generic_private_value_classes_are_caught(
    tmp_path: Path, rule: str, sample: bytes
):
    allowlist = _init_repo(tmp_path, {"notes with spaces.bin": b"prefix\n" + sample + b"\n"})

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert any(
        item.path == "notes with spaces.bin" and item.line == 2 and item.rule == rule
        for item in result.findings
    )


def test_new_credential_rules_require_token_boundaries():
    samples = (
        (
            "github-fine-grained-token",
            _ascii(103, 105, 116, 104, 117, 98, 95, 112, 97, 116, 95)
            + b"A" * 22
            + b"_"
            + b"B" * 59,
        ),
        ("huggingface-token", _ascii(104, 102, 95) + b"C" * 34),
        ("aws-access-key-id", _ascii(65, 75, 73, 65) + b"D" * 16),
    )

    for rule, sample in samples:
        assert rule in {item.rule for item in leak_check.scan_bytes("x", b"(" + sample + b")")}
        for boundary in (b"A", b"a", b"_"):
            assert rule not in {
                item.rule for item in leak_check.scan_bytes("x", boundary + sample)
            }
            assert rule not in {
                item.rule for item in leak_check.scan_bytes("x", sample + boundary)
            }


def test_tailnet_cgnat_range_excludes_neighboring_ranges(tmp_path: Path):
    addresses = b"\n".join(
        (
            b"100." + b"64.0.0",
            b"100." + b"72.1.2",
            b"100." + b"127.255.255",
            b"100." + b"63.255.255",
            b"100." + b"128.0.0",
            b"100." + b"72.256.1",
            b"100." + b"72.1.256",
            b"0" + b"100." + b"72.1.2",
            b"100." + b"72.1.255" + b"0",
        )
    )
    allowlist = _init_repo(tmp_path, {"addresses.txt": addresses + b"\n"})

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert _finding_keys(result) == {
        ("addresses.txt", 1, "network-address"),
        ("addresses.txt", 2, "network-address"),
        ("addresses.txt", 3, "network-address"),
    }


def test_private_192_168_range_is_detected_structurally(tmp_path: Path):
    private_prefix = b"192." + b"168."
    addresses = b"\n".join(
        (
            private_prefix + b"0.0",
            private_prefix + b"11.42",
            private_prefix + b"255.255",
            b"192." + b"167.255.255",
            b"192." + b"169.0.0",
            private_prefix + b"256.1",
            b"0" + private_prefix + b"11.42",
            private_prefix + b"11.420",
        )
    )
    allowlist = _init_repo(tmp_path, {"addresses.txt": addresses + b"\n"})

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert _finding_keys(result) == {
        ("addresses.txt", 1, "network-address"),
        ("addresses.txt", 2, "network-address"),
        ("addresses.txt", 3, "network-address"),
    }


def _fixture_allowlist() -> str:
    return """\
version = 1

[[allow]]
id = "gpu"
rule = "gpu-identifier"
path_glob = "fixtures.txt"
value_glob = "GPU-deadbeef*"
expected_matches = 1
reason = "fictional GPU fixture"

[[allow]]
id = "host"
rule = "machine-hostname"
path_glob = "fixtures.txt"
value_glob = "examplebox-*"
expected_matches = 1
reason = "fictional hostname fixture"
"""


def _safe_fixture_values() -> bytes:
    gpu = b"GPU-" + b"deadbeef-0000-0000-0000-000000000000"
    host = b"examplebox-" + b"f00f"
    return b"\n".join((gpu, host)) + b"\n"


def test_rule_scoped_allowlist_and_untracked_files(tmp_path: Path):
    allowlist = _init_repo(
        tmp_path,
        {"fixtures.txt": _safe_fixture_values()},
        allowlist=_fixture_allowlist(),
    )
    untracked = tmp_path / "untracked.txt"
    untracked.write_bytes(b"ghp_" + b"Z" * 36)

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert result.findings == ()
    assert result.scanned_files == 2


def test_allowlist_does_not_authorize_the_same_value_in_another_path(tmp_path: Path):
    safe_host = b"examplebox-" + b"f00f"
    allowlist_text = """\
version = 1
[[allow]]
id = "host"
rule = "machine-hostname"
path_glob = "fixtures.txt"
value_glob = "examplebox-*"
expected_matches = 1
reason = "one scoped fictional hostname"
"""
    allowlist = _init_repo(
        tmp_path,
        {"fixtures.txt": safe_host, "report.txt": safe_host},
        allowlist=allowlist_text,
    )

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert ("report.txt", 1, "machine-hostname") in _finding_keys(result)


@pytest.mark.parametrize(
    "rule",
    (
        "github-token",
        "github-fine-grained-token",
        "huggingface-token",
        "aws-access-key-id",
        "api-key",
        "slack-token",
        "private-key",
    ),
)
def test_allowlist_rejects_credentials(tmp_path: Path, rule: str):
    credential_allowlist = tmp_path / "credential.toml"
    credential_allowlist.write_text(
        f"""\
version = 1
[[allow]]
id = "credential"
rule = "{rule}"
path_glob = "fixtures.txt"
value_glob = "credential-*"
expected_matches = 1
reason = "must be rejected"
""",
        encoding="utf-8",
    )
    with pytest.raises(leak_check.LeakCheckError, match="non-allowlistable rule"):
        leak_check.load_allowlist(credential_allowlist)


def test_allowlist_rejects_path_globs(tmp_path: Path):
    glob_allowlist = tmp_path / "glob.toml"
    glob_allowlist.write_text(
        _fixture_allowlist().replace('path_glob = "fixtures.txt"', 'path_glob = "*"'),
        encoding="utf-8",
    )
    with pytest.raises(leak_check.LeakCheckError, match="exact repository path"):
        leak_check.load_allowlist(glob_allowlist)


def test_fictional_hostname_rule_does_not_match_unrelated_box_words():
    data = b"sandbox-dead toolbox-face checkbox-acde"

    assert not any(item.rule == "machine-hostname" for item in leak_check.scan_bytes("x", data))


def test_scan_reads_modified_worktree_bytes_not_staged_blob(tmp_path: Path):
    allowlist = _init_repo(tmp_path, {"tracked.txt": "clean\n"})
    private_email = b"person" + b"@" + b"company.local"
    (tmp_path / "tracked.txt").write_bytes(b"changed\n" + private_email + b"\n")

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert ("tracked.txt", 2, "email-address") in _finding_keys(result)


def test_invalid_utf8_binary_is_scanned_with_byte_line_numbers(tmp_path: Path):
    private_token = b"xoxp-" + b"1234567890-abcdef"
    allowlist = _init_repo(tmp_path, {"asset.bin": b"\xff\xfe\n" + private_token})

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert ("asset.bin", 2, "slack-token") in _finding_keys(result)


def test_cli_reports_location_and_rule_without_echoing_value(tmp_path: Path, capsys):
    private_email = b"person" + b"@" + b"company.local"
    allowlist = _init_repo(tmp_path, {"secrets.txt": b"safe\n" + private_email + b"\n"})

    rc = leak_check.main(
        ["--root", str(tmp_path), "--allowlist", str(allowlist)]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "secrets.txt:2:" in captured.err
    assert "leak-check[email-address]" in captured.err
    assert "(filename)" not in captured.err
    assert private_email.decode() not in captured.err
    assert captured.out == ""


def test_cli_redacts_a_credential_bearing_filename(tmp_path: Path, capsys):
    private_token = b"ghp_" + b"R" * 12 + b"S" * 24
    relative = os.fsdecode(private_token + b".txt")
    allowlist = _init_repo(tmp_path, {relative: "clean\n"})

    rc = leak_check.main(["--root", str(tmp_path), "--allowlist", str(allowlist)])
    captured = capsys.readouterr()

    assert rc == 1
    assert private_token.decode() not in captured.err
    assert "ghp_" not in captured.err
    assert private_token[-12:].decode() not in captured.err
    assert "[redacted].txt:1:1 (filename): leak-check[github-token]" in captured.err


def test_displayed_absolute_path_redacts_a_non_placeholder_home():
    private_home = b"/home/" + b"project-owner"
    displayed = leak_check._display_path(os.fsdecode(private_home + b"/repo/file.txt"))

    assert os.fsdecode(private_home) not in displayed
    assert displayed.startswith("[redacted]/")


def test_allowlist_count_drift_fails_closed_with_locations(tmp_path: Path):
    safe_host = b"examplebox-" + b"f00f"
    allowlist_text = """\
version = 1
[[allow]]
id = "host"
rule = "machine-hostname"
path_glob = "fixtures.txt"
value_glob = "examplebox-*"
expected_matches = 1
reason = "one fictional hostname"
"""
    allowlist = _init_repo(
        tmp_path,
        {"fixtures.txt": safe_host + b"\n" + safe_host + b"\n"},
        allowlist=allowlist_text,
    )

    with pytest.raises(leak_check.LeakCheckError, match="matched 2 candidate") as exc_info:
        leak_check.scan_repository(tmp_path, allowlist)

    assert "fixtures.txt:1:" in str(exc_info.value)
    assert "fixtures.txt:2:" in str(exc_info.value)


def test_allowlist_identifier_is_not_echoed_by_count_errors(tmp_path: Path):
    private_id = "private-entry"
    safe_host = b"examplebox-" + b"f00f"
    allowlist_text = f"""\
version = 1
[[allow]]
id = "{private_id}"
rule = "machine-hostname"
path_glob = "fixtures.txt"
value_glob = "examplebox-*"
expected_matches = 2
reason = "force a count error"
"""
    allowlist = _init_repo(
        tmp_path,
        {"fixtures.txt": safe_host},
        allowlist=allowlist_text,
    )

    with pytest.raises(leak_check.LeakCheckError, match="allowlist entry 1") as exc_info:
        leak_check.scan_repository(tmp_path, allowlist)

    assert private_id not in str(exc_info.value)


@pytest.mark.parametrize("version", ("true", "1.0", '"1"'))
def test_allowlist_version_requires_exact_integer_one(tmp_path: Path, version: str):
    path = tmp_path / "allowlist.toml"
    path.write_text(f"version = {version}\nallow = []\n", encoding="utf-8")

    with pytest.raises(leak_check.LeakCheckError, match="unsupported version; set version = 1"):
        leak_check.load_allowlist(path)


def test_non_utf8_allowlist_uses_controlled_cli_error(tmp_path: Path, capsys):
    allowlist = _init_repo(tmp_path, {"clean.txt": "clean\n"})
    allowlist.write_bytes(b"\xff\xfe")

    rc = leak_check.main(["--root", str(tmp_path), "--allowlist", str(allowlist)])
    captured = capsys.readouterr()

    assert rc == 2
    assert "allowlist must be UTF-8 text" in captured.err
    assert "Traceback" not in captured.err


def test_symlink_target_is_scanned_without_following_external_file(tmp_path: Path):
    outside = tmp_path.parent / "external-symlink-target.txt"
    outside.write_bytes(b"external fixture payload\n")
    link = tmp_path / "linked.txt"
    os.symlink(outside, link)
    allowlist = _init_repo(tmp_path, {})
    subprocess.run(
        ["git", "add", "linked.txt"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    result = leak_check.scan_repository(tmp_path, allowlist)

    assert result.findings == ()


def test_missing_tracked_file_is_an_error(tmp_path: Path):
    allowlist = _init_repo(tmp_path, {"gone.txt": "clean\n"})
    (tmp_path / "gone.txt").unlink()

    with pytest.raises(leak_check.LeakCheckError, match="cannot read tracked path"):
        leak_check.scan_repository(tmp_path, allowlist)
