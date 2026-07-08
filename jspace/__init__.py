"""
jspace-obfuscation — Does the global workspace survive obfuscation pressure?

Tests whether the J-space (Jacobian-lens) oversight channel is robust to the
optimization pressure that collapses chain-of-thought (CoT) text monitoring.

See DEPENDENCIES.md for the verified facts this project is built on (the paper
uses Claude; the open-weights target is Qwen3.6-27B; no pre-fit lens ships, so
the lens must be fitted).
"""

from .types import Readout, Trajectory, Item, CType

__all__ = ["Readout", "Trajectory", "Item", "CType"]
