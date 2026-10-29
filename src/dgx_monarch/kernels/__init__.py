"""Kernel replacements DGX Monarch installs in its own processes.

A replacement is admitted only when a dated record proves it exact against the
kernel it replaces, and each one has an off switch and falls back to the stock
kernel instead of refusing (docs/DESIGN.md, section 4, "Kernel replacements").
"""
