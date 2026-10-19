"""Fixed local-only CLI adapters when Studio has no selected cluster config."""

from __future__ import annotations

import argparse
import math


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("doctor", "reap.preview", "reap.apply"))
    parser.add_argument("--grace", type=float, default=0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.grace) or args.grace < 0:
        parser.error("grace must be finite and nonnegative")
    if args.action == "doctor":
        from ..cli.doctor import doctor_exit

        return doctor_exit(None, as_json=True)
    from ..cli.actor_sweep import reap_command

    return 0 if reap_command(None, dry_run=args.action == "reap.preview", grace_s=args.grace) else 1


if __name__ == "__main__":
    raise SystemExit(main())
