"""Shared helpers for reading concept scores out of a lens."""
from __future__ import annotations

from typing import Sequence


def cluster_token_ids(tok, cluster: Sequence[str]) -> list[int]:
    """First-token ids for each surface form across casing / leading-space variants.

    Lens vocab tokens are typically space-prefixed (' Italy'), so we cover the
    bare, space-prefixed, capitalized, and space+capitalized forms and keep the
    first token of each (the token the model would be 'poised' to start C with).
    """
    ids = set()
    for form in cluster:
        variants = {form, " " + form, form.capitalize(), " " + form.capitalize(),
                    form.upper(), " " + form.upper()}
        for v in variants:
            enc = tok(v, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(int(enc[0]))
    return sorted(ids)
