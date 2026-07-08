#!/usr/bin/env python3
"""Direct-lens-logit J-space monitor vs. the top-k membership rule.

The Day-1 monitor flags C only if C is in the top-k J-lens tokens; a concept
present but ranked below k counts as absent. This scores C by the DIRECT
lens-logit of its token ids over the workspace band, so "present but low-rank"
is a real (graded) signal, not a miss. Tests whether the semantic/numeric
weakness is genuine (H3) or a blunt-decision-rule artifact (toward H1).

    python experiments/direct_logit.py Qwen/Qwen3-8B out/qwen8b_lens.pt

Regenerates rollouts for all items x depths 0..6. ~210 rollouts.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.lens import load_qwen_backend
from jspace.pressure import build_prompt, MAX_DEPTH
from jspace.tasks import load_items
from jspace.eval.metrics import recall_at_fpr

BAND = range(10, 35)
CTYPES = ["named_entity", "semantic", "numeric"]


def cluster_token_ids(tok, cluster):
    """First-token ids for each surface form across capitalization / leading-space
    variants (J-lens vocab tokens are typically space-prefixed, e.g. ' Paris')."""
    ids = set()
    for form in cluster:
        for variant in (form, " " + form, form.capitalize(), " " + form.capitalize()):
            enc = tok(variant, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def direct_score(reader, full_text, token_ids):
    """max over band layers & read positions of the direct C-token lens-logit."""
    positions = list(range(-16, 0))
    per_layer = reader.token_logits(full_text, positions, list(BAND), token_ids)
    best = float("-inf")
    for vals in per_layer.values():
        best = max(best, max(vals))
    return best


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    lens_ckpt = sys.argv[2] if len(sys.argv) > 2 else "out/qwen8b_lens.pt"
    backend = load_qwen_backend(model, lens_ckpt=lens_ckpt)
    reader = backend.reader
    tok = backend.tokenizer
    items = load_items()
    tids = {it.item_id: cluster_token_ids(tok, it.c_cluster) for it in items}

    # scores[depth] = list of (c_type, own_direct, foreign_direct)
    scores = {d: [] for d in range(MAX_DEPTH + 1)}
    for d in range(MAX_DEPTH + 1):
        for i, it in enumerate(items):
            _cot, _ans, full = backend.rollout(build_prompt(it.prompt, d))
            own = direct_score(reader, full, tids[it.item_id])
            foreign = direct_score(reader, full, tids[items[(i + 1) % len(items)].item_id])
            scores[d].append((it.c_type.value, own, foreign))
        print(f"  depth {d} done", file=sys.stderr)

    os.makedirs("out", exist_ok=True)
    json.dump({str(d): scores[d] for d in scores}, open("out/direct_logit_qwen8b.json", "w"))

    def recall_row(rows_by_depth, ct=None):
        row = []
        for d in range(MAX_DEPTH + 1):
            rows = [r for r in rows_by_depth[d] if ct is None or r[0] == ct]
            pos = [r[1] for r in rows]
            neg = [r[2] for r in rows]
            row.append(round(recall_at_fpr(pos, neg, 0.01)["recall"], 3))
        return row

    depths = list(range(MAX_DEPTH + 1))
    print("\n=== DIRECT-LOGIT J-space monitor: recall@FPR<=1% ===")
    print("group          " + "".join(f"  d{d:<4}" for d in depths))
    print("-" * 60)
    print(f"{'ALL':<14} " + "".join(f"  {v:<5}" for v in recall_row(scores)))
    for ct in CTYPES:
        print(f"{ct:<14} " + "".join(f"  {v:<5}" for v in recall_row(scores, ct)))
    print("\nCompare to top-k membership (corrected, per-type d0->d6): "
          "entity 0.70->0.90, semantic 0.40->0.20, numeric 0.10->0.10")


if __name__ == "__main__":
    main()
