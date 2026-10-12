"""Small configuration readers that do not import Monarch or Torch."""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

from .config_schema import ClusterConfigError, validate_worker_args
from .monarch_policy import MonarchPolicy, parse_policy


def local_settings(path: Path | None) -> tuple[dict[str, object], MonarchPolicy]:
    """Read local actor knobs once; malformed supplied tables fail visibly."""
    if path is None:
        return {}, MonarchPolicy()
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ClusterConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError:
        return {}, MonarchPolicy()
    args = validate_worker_args(raw.get("worker_args", {}), context=f"{path}: worker_args")
    return dict(args), parse_policy(raw.get("monarch", {}))


def trust_config_fingerprint(config, fallback: str) -> str:
    """Bind effective runtime settings without invalidating trust for export knobs."""
    from .config import ClusterConfig

    if not isinstance(config, ClusterConfig):
        return fallback
    payload = asdict(config)
    payload["monarch"].pop("diagnostics")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
