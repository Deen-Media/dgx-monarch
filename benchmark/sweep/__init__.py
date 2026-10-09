"""Derive sweep cells from a local TOML configuration and record their outcomes.

Entry points (each accepts ``--help``):

    python -m benchmark.sweep.convert     templates to API graphs
    python -m benchmark.sweep.matrix      derive and label cells
    python -m benchmark.sweep.run         run a session
    python -m benchmark.sweep.report      summarize results and findings
    python -m benchmark.sweep.memsample   sample meminfo at 1 Hz
    python -m benchmark.sweep.memhog      hold MemAvailable at a target

Labels use runtime guards. Records stay in the operator's output directory;
exported text is sanitized. The last two tools support manually driven
capacity tests between sweep waves. Their memory records use the runner's
fields and JSON-lines format.
"""
