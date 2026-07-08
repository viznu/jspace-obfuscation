"""Shared data model for tasks, trajectories, and lens readouts.

Kept dependency-free (stdlib only) so the whole analysis spine runs and is
testable on a laptop with no torch / no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CType(str, Enum):
    """Type of the forbidden intermediate concept C.

    Tests H3 (concept-dependent robustness): the paper hints number-like
    intermediates may be computed *outside* the workspace, so we expect NUMERIC
    C to escape the J-space monitor more easily than a NAMED_ENTITY / SEMANTIC C.
    """
    NAMED_ENTITY = "named_entity"   # e.g. C = "Italy" in a capital->country->currency hop
    SEMANTIC = "semantic"           # e.g. C = a category / property word ("primate")
    NUMERIC = "numeric"             # e.g. C = an arithmetic intermediate (56)


@dataclass
class Item:
    """A forbidden-intermediate task item (Setting A).

    Solving `prompt` provably requires computing the intermediate concept C.
    The "misbehavior" analog: produce `answer` correctly while never naming any
    token of `c_cluster` in the CoT.
    """
    item_id: str
    prompt: str
    answer: str                     # the correct final answer (for grading task success)
    c_type: CType
    c_canonical: str                # the human-readable name of C, e.g. "Italy"
    c_cluster: list[str]            # C + synonyms/aliases/surface forms, lowercased
    # An answer-matched item where C is NOT required — the negative control
    # (monitors should NOT fire on these; §5.5 load-bearing verification).
    control_prompt: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Readout:
    """A single J-lens top-k readout at one (layer, position).

    `tokens` are decoded vocabulary strings (already stripped/lowercased for
    matching); `weights` are the corresponding lens logits or normalized scores,
    aligned index-for-index with `tokens`.
    """
    layer: int
    position: int                   # absolute or negative index into the sequence
    tokens: list[str]
    weights: list[float]

    def weight_of(self, token: str) -> float:
        """Max weight assigned to `token` in this readout, or -inf if absent."""
        tok = token.lower()
        best = float("-inf")
        for t, w in zip(self.tokens, self.weights):
            if t == tok:
                best = max(best, w)
        return best


@dataclass
class Trajectory:
    """One model rollout on one item, plus everything the monitors need to see."""
    item_id: str
    prompt: str
    cot: str                        # the reasoning text (CoT-only monitor sees this)
    answer: str                     # the final answer text (output-only monitor sees this)
    c_cluster: list[str]            # target concept surface forms for THIS item (lowercased)
    is_positive: bool               # ground truth: C is load-bearing / misbehavior present
    readouts: list[Readout] = field(default_factory=list)  # J-lens readouts (workspace band)
    task_success: Optional[bool] = None   # did `answer` match the gold answer?
    meta: dict = field(default_factory=dict)
