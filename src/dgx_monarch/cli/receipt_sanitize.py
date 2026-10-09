"""Disclosure-safe operator-receipt note sanitization."""
from __future__ import annotations

import ipaddress
import re

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_COMMAND = re.compile(r"\b(?:argv|command|cmd)\s*[:=]\s*.*$", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:password|passwd|secret|token|credential|api[_-]?key|ssh[_-]?key|"
    r"private[_-]?key|authorization)\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
_KNOWN_SECRET = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|"
    r"sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/-]{8,})\b",
    re.IGNORECASE,
)
_ENV_ASSIGNMENT = re.compile(
    r"\b[A-Z][A-Z0-9_]{2,}\s*=\s*(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_KEYED_HOST = re.compile(
    r"\b(?:host(?:name)?|node|peer|address|endpoint)\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_HOSTLIKE = re.compile(
    r"\b(?:worker|spark|dgx|node|host)[_-][A-Za-z0-9._-]+\b", re.IGNORECASE
)
_FQDN = re.compile(
    r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}\b"
)
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9])(?:~|/)(?:[^\s,;:'\"()]+/?)+")
_RELATIVE_PATH = re.compile(
    r"(?<![A-Za-z0-9._~/])(?:\.\.?/)?(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+/?"
)
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\[^\s,;:'\"()]+")
_EXCEPTION_REPR = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure)\s*\([^)]*\)"
)
_IP_CANDIDATE = re.compile(
    r"(?<![0-9A-Fa-f:.])(?:\d{1,3}(?:\.\d{1,3}){3}|"
    r"[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,}"
    r"(?:%[A-Za-z0-9_.-]+)?)(?![0-9A-Fa-f:.])"
)
_SAFE_DOTTED = frozenset({
    "dgx_monarch.cli", "torch.bfloat16", "torch.float16", "torch.float32", "torch.nn",
})


def _is_safe_dotted(candidate: str) -> bool:
    return candidate in _SAFE_DOTTED


def _redact_ip(match: re.Match[str]) -> str:
    candidate = match.group(0)
    try:
        ipaddress.ip_address(candidate.split("%", 1)[0])
    except ValueError:
        return candidate
    return "[network identity redacted]"


def _redact_fqdn(match: re.Match[str]) -> str:
    """Preserve explicit file-like tokens while redacting arbitrary hosts."""
    candidate = match.group(0)
    labels = candidate.lower().split(".")
    benign_file_suffixes = {
        "csv", "json", "log", "md", "py", "safetensors", "toml", "txt", "yaml", "yml",
    }
    host_prefixes = ("dgx", "host", "node", "spark", "worker")
    if _is_safe_dotted(candidate):
        return candidate
    if (
        len(labels) == 2
        and labels[-1] in benign_file_suffixes
        and not labels[0].startswith(host_prefixes)
    ):
        return candidate
    return "[network identity redacted]"


def _redact_hostlike(match: re.Match[str]) -> str:
    candidate = match.group(0)
    return candidate if _is_safe_dotted(candidate) else "[network identity redacted]"


def sanitize_note(value: str) -> str:
    """Return one bounded, single-line note with disclosure risks removed."""
    if not isinstance(value, str):
        raise TypeError("receipt notes must be strings, not exception or metadata objects")
    text = _CONTROL.sub(" ", value).strip()
    text = _COMMAND.sub("[command redacted]", text)
    text = _URL.sub("[url redacted]", text)
    text = _SECRET_ASSIGNMENT.sub("[secret redacted]", text)
    text = _KNOWN_SECRET.sub("[secret redacted]", text)
    text = _ENV_ASSIGNMENT.sub("[environment redacted]", text)
    text = _KEYED_HOST.sub("[network identity redacted]", text)
    text = _IP_CANDIDATE.sub(_redact_ip, text)
    text = _RELATIVE_PATH.sub("[path redacted]", text)
    text = _FQDN.sub(_redact_fqdn, text)
    text = _HOSTLIKE.sub(_redact_hostlike, text)
    text = _WINDOWS_PATH.sub("[path redacted]", text)
    text = _POSIX_PATH.sub("[path redacted]", text)
    text = _EXCEPTION_REPR.sub("[exception redacted]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "[redacted]"
    if len(text) > 240:
        text = f"{text[:227].rstrip()} [truncated]"
    return text
