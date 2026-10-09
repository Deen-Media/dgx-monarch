"""`python -m dgx_monarch` == `dgxm` (DESIGN.md §6.1): the fallback for venvs
whose pip script directory is not on PATH (ComfyUI portable installs)."""
from .cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
