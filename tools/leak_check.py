#!/usr/bin/env python3
"""Fail closed when tracked working-tree files contain private identifiers.

The checker reads the working-tree copy of every path ``git ls-files`` lists, so
uncommitted edits are scanned and untracked files are not. It does not inspect
index blobs, skip binary files, or print the matched value: CI diagnostics
identify only the path, line, column, and rule.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import stat
import subprocess
import sys
import tomllib
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


class LeakCheckError(RuntimeError):
    """The repository or allowlist could not be scanned safely."""


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[bytes]
    description: str


# Rules match the shape of a private identifier, not its value, so the scanner
# itself holds no private user name, host name or key file name. Home paths
# under these placeholder account names pass; tests and examples use them.
_PLACEHOLDER_ACCOUNT = rb"(?:alice|example|operator|runner|u|user|worker)"
_ACCOUNT_BOUNDARY = rb"(?:[^A-Za-z0-9._-]|$)"
_PRIVATE_HOME_PATH = re.compile(
    rb"(?<![A-Za-z0-9._-])(?:"
    rb"/root(?=/)"
    rb"|/home/(?!" + _PLACEHOLDER_ACCOUNT + _ACCOUNT_BOUNDARY + rb")[A-Za-z0-9._-]+"
    rb"|/Users/(?!" + _PLACEHOLDER_ACCOUNT + _ACCOUNT_BOUNDARY + rb")[A-Za-z0-9._-]+"
    rb"|(?:[A-Za-z]:[\\/]|\\\\[A-Za-z0-9._-]+\\)Users[\\/](?!"
    + _PLACEHOLDER_ACCOUNT
    + _ACCOUNT_BOUNDARY
    + rb")[A-Za-z0-9._-]+"
    rb")",
    re.IGNORECASE,
)
_RIG_HOSTNAME = re.compile(
    rb"(?<![A-Za-z0-9])(?:dgx|edge|spark)[A-Za-z0-9-]*?"
    rb"(?=[0-9a-f]{4,}(?![A-Za-z0-9]))(?=[0-9a-f]*[0-9])[0-9a-f]{4,}"
    rb"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CUSTOM_PRIVATE_KEY_FILENAME = re.compile(
    rb"(?<![A-Za-z0-9])id_(?:rsa|dsa|ecdsa|ed25519)"
    rb"(?:[_-][A-Za-z0-9.-]+)+(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_IPV4_OCTET = rb"(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})"
_NETWORK_ADDRESS = re.compile(
    rb"(?<![0-9])(?:"
    rb"100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\."
    + _IPV4_OCTET
    + rb"|192\.168\."
    + _IPV4_OCTET
    + rb")"
    + rb"\."
    + _IPV4_OCTET
    + rb"(?![0-9])"
)
_PEM_PRIVATE_KEY = re.compile(
    b"-----BEGIN "
    + rb"(?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?"
    + b"PRIVATE KEY-----"
)
_RESERVED_EMAIL_DOMAIN = (
    rb"(?:"
    rb"(?:[A-Za-z0-9-]+\.)*example\.(?:com|net|org)"
    rb"|(?:[A-Za-z0-9-]+\.)+(?:example|invalid|test)"
    rb")"
)

RULES = (
    Rule(
        "rig-hostname",
        _RIG_HOSTNAME,
        "rig-style hostname with a hexadecimal suffix",
    ),
    Rule(
        "private-home",
        _PRIVATE_HOME_PATH,
        "home path of an account that is not a placeholder",
    ),
    Rule(
        "key-filename",
        _CUSTOM_PRIVATE_KEY_FILENAME,
        "SSH private-key filename with a custom suffix",
    ),
    Rule(
        "network-address",
        _NETWORK_ADDRESS,
        "IPv4 address in 192.168 or the CGNAT range",
    ),
    Rule(
        "machine-hostname",
        re.compile(
            rb"examplebox-[0-9A-Fa-f]{4}",
            re.IGNORECASE,
        ),
        "fictional fixture hostname outside the allowlist",
    ),
    Rule(
        "email-address",
        re.compile(
            rb"(?<![A-Za-z0-9._%+-])"
            rb"[A-Za-z0-9._%+-]+@"
            rb"(?!" + _RESERVED_EMAIL_DOMAIN + rb"(?![A-Za-z0-9.-]))"
            rb"[A-Za-z0-9.-]+\.[A-Za-z]{2,63}"
            rb"(?![A-Za-z0-9._%+-])",
            re.IGNORECASE,
        ),
        "email address outside the reserved example and test domains",
    ),
    Rule(
        "github-token",
        re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
        "GitHub credential token",
    ),
    Rule(
        "github-fine-grained-token",
        re.compile(
            rb"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9]{22}_"
            rb"[A-Za-z0-9]{59}"
            rb"(?![A-Za-z0-9_])"
        ),
        "GitHub fine-grained credential token",
    ),
    Rule(
        "huggingface-token",
        re.compile(
            rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34}(?![A-Za-z0-9_])"
        ),
        "Hugging Face credential token",
    ),
    Rule(
        "aws-access-key-id",
        re.compile(
            rb"(?<![A-Za-z0-9_])AKIA[A-Z0-9]{16}(?![A-Za-z0-9_])"
        ),
        "AWS access-key identifier",
    ),
    Rule(
        "api-key",
        re.compile(
            b"sk-" + rb"(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"
        ),
        "API credential token",
    ),
    Rule(
        "slack-token",
        re.compile(rb"xox[a-z]-[A-Za-z0-9-]{10,}"),
        "Slack credential token",
    ),
    Rule("private-key", _PEM_PRIVATE_KEY, "PEM private-key header"),
    Rule(
        "gpu-identifier",
        # The dash after the eight hex digits lets a bare fixture prefix such as `GPU-deadbeef` pass.
        re.compile(
            rb"GPU-[0-9A-Fa-f]{8}-[A-Za-z0-9-]*",
            re.IGNORECASE,
        ),
        "GPU UUID-shaped identifier",
    ),
)
_RULES_BY_NAME = {rule.name: rule for rule in RULES}
_APPROVED_ALLOWLIST_VALUES = {
    "gpu-identifier": frozenset({"GPU-deadbeef*"}),
    "machine-hostname": frozenset({"examplebox-*"}),
}


@dataclass(frozen=True, repr=False)
class Allowance:
    ordinal: int
    identifier: str
    rule: str
    path_glob: str
    value_glob: str
    expected_matches: int
    reason: str


@dataclass(frozen=True)
class Finding:
    path: str = field(repr=False)
    line: int
    column: int
    rule: str
    description: str
    matched: bytes = field(repr=False, compare=False)
    is_filename: bool = False

    @property
    def location(self) -> str:
        marker = " (filename)" if self.is_filename else ""
        return f"{_display_path(self.path)}:{self.line}:{self.column}{marker}"


@dataclass(frozen=True)
class ScanResult:
    findings: tuple[Finding, ...]
    scanned_files: int


def _display_path(value: str) -> str:
    """Redact rule matches and escape controls before a path reaches CI logs."""
    raw = os.fsencode(value)
    spans: list[tuple[int, int]] = []
    for rule in RULES:
        spans.extend((match.start(), match.end()) for match in rule.pattern.finditer(raw))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if merged:
        pieces: list[bytes] = []
        cursor = 0
        for start, end in merged:
            pieces.extend((raw[cursor:start], b"[redacted]"))
            cursor = end
        pieces.append(raw[cursor:])
        raw = b"".join(pieces)
    decoded = os.fsdecode(raw)
    return decoded.encode("unicode_escape", "backslashreplace").decode("ascii")


def _allowlist_error(path: Path, detail: str) -> LeakCheckError:
    return LeakCheckError(f"invalid allowlist {_display_path(str(path))}: {detail}")


def load_allowlist(path: Path) -> tuple[Allowance, ...]:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LeakCheckError(
            f"cannot read allowlist {_display_path(str(path))}: {type(exc).__name__}"
        ) from exc
    except UnicodeError as exc:
        raise _allowlist_error(path, "allowlist must be UTF-8 text") from exc
    except tomllib.TOMLDecodeError as exc:
        raise _allowlist_error(path, "malformed TOML") from exc

    if set(payload) != {"version", "allow"}:
        raise _allowlist_error(path, "top-level keys must be exactly version and allow")
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise _allowlist_error(path, "unsupported version; set version = 1")
    raw_entries = payload["allow"]
    if not isinstance(raw_entries, list):
        raise _allowlist_error(path, "allow must be an array of tables")

    required = {
        "id",
        "rule",
        "path_glob",
        "value_glob",
        "expected_matches",
        "reason",
    }
    entries: list[Allowance] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_entries, 1):
        if not isinstance(raw, dict) or set(raw) != required:
            raise _allowlist_error(
                path, f"allow entry {index} must be a table with exactly the keys id, rule, path_glob, "
                "value_glob, expected_matches and reason"
            )
        if not all(isinstance(raw[key], str) for key in required - {"expected_matches"}):
            raise _allowlist_error(path, f"allow entry {index} has a non-string field")
        expected = raw["expected_matches"]
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise _allowlist_error(path, f"allow entry {index} expected_matches must be a positive integer")
        identifier = raw["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", identifier):
            raise _allowlist_error(
                path, f"allow entry {index} id must be lowercase letters, digits and dashes, not starting with a dash"
            )
        if identifier in identifiers:
            raise _allowlist_error(path, f"allow entry {index} duplicates an earlier id")
        identifiers.add(identifier)
        rule_name = raw["rule"]
        if rule_name not in _RULES_BY_NAME:
            raise _allowlist_error(path, f"allow entry {index} names an unknown rule")
        approved_values = _APPROVED_ALLOWLIST_VALUES.get(rule_name)
        if approved_values is None:
            raise _allowlist_error(path, f"allow entry {index} names a non-allowlistable rule")
        path_glob = raw["path_glob"]
        value_glob = raw["value_glob"]
        reason = raw["reason"].strip()
        parsed_path = PurePosixPath(path_glob)
        if (
            not path_glob
            or "\n" in path_glob
            or "\r" in path_glob
            or "\\" in path_glob
            or any(char in path_glob for char in "*?[]")
            or parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or parsed_path.as_posix() != path_glob
        ):
            raise _allowlist_error(
                path, f"allow entry {index} path_glob must be an exact repository path"
            )
        literal_value = re.sub(r"[?*\[\]]", "", value_glob)
        if len(literal_value) < 4 or "\n" in value_glob or "\r" in value_glob:
            raise _allowlist_error(
                path, f"allow entry {index} value_glob needs four or more literal characters on one line"
            )
        if value_glob not in approved_values:
            raise _allowlist_error(path, f"allow entry {index} value_glob is not approved")
        if not reason:
            raise _allowlist_error(path, f"allow entry {index} needs a reason")
        entries.append(
            Allowance(
                ordinal=index,
                identifier=identifier,
                rule=rule_name,
                path_glob=path_glob,
                value_glob=value_glob,
                expected_matches=expected,
                reason=reason,
            )
        )
    return tuple(entries)


def _git_root(candidate: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise LeakCheckError("cannot resolve a Git working tree")
    return Path(os.fsdecode(result.stdout.rstrip(b"\n"))).resolve()


def tracked_paths(root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise LeakCheckError("git ls-files failed")

    paths: list[str] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise LeakCheckError("git ls-files returned an unsafe path")
        paths.append(relative)
    return tuple(sorted(set(paths)))


def _read_tracked_file(root: Path, relative: str) -> bytes:
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            return os.fsencode(os.readlink(path))
        if not stat.S_ISREG(mode):
            raise LeakCheckError(
                f"tracked path {_display_path(relative)} is not a regular file or symlink"
            )
        return path.read_bytes()
    except LeakCheckError:
        raise
    except OSError as exc:
        raise LeakCheckError(
            f"cannot read tracked path {_display_path(relative)}: {type(exc).__name__}"
        ) from exc


def _line_starts(data: bytes) -> list[int]:
    return [0, *(index + 1 for index, value in enumerate(data) if value == 10)]


def _line_column(line_starts: list[int], offset: int) -> tuple[int, int]:
    line_index = bisect_right(line_starts, offset) - 1
    return line_index + 1, offset - line_starts[line_index] + 1


def scan_bytes(relative: str, data: bytes) -> list[Finding]:
    candidates: list[Finding] = []
    path_bytes = os.fsencode(relative)
    line_starts = _line_starts(data)
    for rule in RULES:
        for match in rule.pattern.finditer(path_bytes):
            candidates.append(
                Finding(
                    path=relative,
                    line=1,
                    column=match.start() + 1,
                    rule=rule.name,
                    description=rule.description,
                    matched=match.group(),
                    is_filename=True,
                )
            )
        for match in rule.pattern.finditer(data):
            line, column = _line_column(line_starts, match.start())
            candidates.append(
                Finding(
                    path=relative,
                    line=line,
                    column=column,
                    rule=rule.name,
                    description=rule.description,
                    matched=match.group(),
                )
            )
    return candidates


def _matches_allowance(finding: Finding, allowance: Allowance) -> bool:
    if finding.rule != allowance.rule:
        return False
    value = finding.matched.decode("ascii")
    return finding.path == allowance.path_glob and fnmatch.fnmatchcase(
        value, allowance.value_glob
    )


def _apply_allowlist(
    candidates: list[Finding], allowances: tuple[Allowance, ...]
) -> tuple[Finding, ...]:
    observed: dict[int, list[Finding]] = {item.ordinal: [] for item in allowances}
    findings: list[Finding] = []
    for candidate in candidates:
        matches = [item for item in allowances if _matches_allowance(candidate, item)]
        if len(matches) > 1:
            ordinals = ", ".join(str(item.ordinal) for item in matches)
            raise LeakCheckError(
                f"overlapping allowlist entries ({ordinals}) at {candidate.location}"
            )
        if not matches:
            findings.append(candidate)
            continue
        observed[matches[0].ordinal].append(candidate)

    for allowance in allowances:
        matches = observed[allowance.ordinal]
        if len(matches) != allowance.expected_matches:
            locations = ", ".join(item.location for item in matches) or "none"
            raise LeakCheckError(
                f"allowlist entry {allowance.ordinal} matched {len(matches)} candidate(s), "
                f"expected {allowance.expected_matches}; locations: {locations}"
            )
    return tuple(sorted(findings, key=lambda item: (item.path, item.line, item.column, item.rule)))


def scan_repository(root: Path, allowlist_path: Path) -> ScanResult:
    git_root = _git_root(root)
    allowances = load_allowlist(allowlist_path)
    paths = tracked_paths(git_root)
    candidates: list[Finding] = []
    for relative in paths:
        candidates.extend(scan_bytes(relative, _read_tracked_file(git_root, relative)))
    return ScanResult(_apply_allowlist(candidates, allowances), len(paths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="scan tracked working-tree files for private identifiers and credentials"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="path inside the Git working tree (default: repository containing this script)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="allowlist TOML (default: <root>/tools/leak_allowlist.toml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = _git_root(args.root)
        allowlist = args.allowlist or root / "tools" / "leak_allowlist.toml"
        if not allowlist.is_absolute():
            allowlist = root / allowlist
        result = scan_repository(root, allowlist)
    except LeakCheckError as exc:
        print(f"leak check error: {exc}", file=sys.stderr)
        return 2

    for finding in result.findings:
        print(
            f"{finding.location}: leak-check[{finding.rule}]: {finding.description}",
            file=sys.stderr,
        )
    if result.findings:
        print(
            f"leak check failed: {len(result.findings)} finding(s) in "
            f"{result.scanned_files} tracked file(s)",
            file=sys.stderr,
        )
        return 1
    print(f"leak check passed: {result.scanned_files} tracked file(s) scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
