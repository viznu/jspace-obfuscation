#!/usr/bin/env python3
"""Does the Jacobian lens beat the plain logit lens for a SILENT representation
of the bridge entity, in MIDDLE layers (below L_deg)?

This is the make-or-break comparison. Jacobian vs logit only
(they share the code path); the tuned-lens and supervised-probe arms are added
next, and only matter if J already beats plain logit here.

  python experiments/jacobian_vs_logit_sweep.py Qwen/Qwen3-8B out/qwen8b_lens150.pt
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.lens import load_qwen_backend
from jspace.lens.util import cluster_token_ids
from jspace.tasks.counterfactual import build_counterfactuals, prompt_for
from jspace.eval.metrics import (auroc, bootstrap_delta_auroc, permutation_delta_pvalue,
                                 recall_above_max_negative)

L_DEG_CORR = 0.95        # arms "degenerate" where full-vocab readouts correlate this much


def measure_l_deg(reader, layers, probe_texts):
    """L_deg = lowest layer where the Jacobian and logit full-vocab readouts
    correlate >= L_DEG_CORR (averaged over probe positions). Above L_deg the lenses
    are ~the same, so a Jacobian 'win' there is degeneracy, not a workspace."""
    import torch

    per_layer = {L: [] for L in layers}
    for text in probe_texts:
        j, _, _ = reader.lens.apply(reader.model, text, layers=list(layers),
                                    positions=[-1], use_jacobian=True)
        l, _, _ = reader.lens.apply(reader.model, text, layers=list(layers),
                                    positions=[-1], use_jacobian=False)
        for L in layers:
            a = j[L].float().flatten()
            b = l[L].float().flatten()
            a = a - a.mean(); b = b - b.mean()
            corr = float((a @ b) / (a.norm() * b.norm() + 1e-9))
            per_layer[L].append(corr)
    mean_corr = {L: sum(v) / len(v) for L, v in per_layer.items()}
    l_deg = next((L for L in layers if mean_corr[L] >= L_DEG_CORR), max(layers))
    return l_deg, mean_corr


def anchor_a_text(backend, item):
    """Silent answer-commit read: no-CoT rollout, then the text truncated right
    before the gold answer token. Returns (anchor_text, success, silent) or None."""
    _cot, ans, full = backend.rollout(prompt_for(item, "no_cot"))
    success = item.answer.lower() in ans.lower()
    low, g = full.lower(), item.answer.lower()
    idx = low.rfind(g)                       # the emitted answer (last occurrence)
    if idx <= 0:
        return None                          # answer not localizable -> attrition
    anchor = full[:idx].rstrip()
    silent = not any(c in anchor.lower() for c in item.target_cluster)
    return anchor, success, silent


def scores_at(reader, tok, text, layers, cluster):
    tids = cluster_token_ids(tok, cluster)
    j = reader.token_logprobs(text, [-1], layers, tids, use_jacobian=True, reduce="mean")
    l = reader.token_logprobs(text, [-1], layers, tids, use_jacobian=False, reduce="mean")
    return {L: j[L][0] for L in layers}, {L: l[L][0] for L in layers}


def per_layer_table(pos_j, pos_l, neg_j, neg_l, layers, l_deg):
    """pos_j[layer] = list of per-item J log-probs for positives, etc."""
    rows = []
    for L in layers:
        pj, pl = pos_j[L], pos_l[L]
        nj, nl = neg_j[L], neg_l[L]
        aj, al = auroc(pj, nj), auroc(pl, nl)
        point, lo, hi = bootstrap_delta_auroc(pj, nj, pl, nl, n_boot=2000, seed=0)
        p = permutation_delta_pvalue(pj, nj, pl, nl, n_perm=2000, seed=0)
        rows.append(dict(layer=L, auroc_j=round(aj, 3), auroc_logit=round(al, 3),
                         delta=round(point, 3), lo=round(lo, 3), hi=round(hi, 3),
                         p=round(p, 3), rmn_j=round(recall_above_max_negative(pj, nj), 3),
                         band="mid" if L < l_deg else "late"))
    return rows


def collect(backend, reader, tok, items, layers, anchor):
    pos_j = {L: [] for L in layers}; pos_l = {L: [] for L in layers}
    neg_j = {L: [] for L in layers}; neg_l = {L: [] for L in layers}
    kept = {"pos": 0, "neg": 0}; attrition = {"no_anchor": 0, "leaked": 0}
    for it in items:
        if anchor == "A":
            res = anchor_a_text(backend, it)
            if res is None:
                attrition["no_anchor"] += 1; continue
            text, success, silent = res
            if not (success and silent):
                attrition["leaked" if not silent else "no_anchor"] += 1; continue
        else:  # anchor P: last prompt token, no generation
            text = prompt_for(it, "no_cot")
        j, l = scores_at(reader, tok, text, layers, it.target_cluster)
        tgt = (pos_j, pos_l) if it.should_fire else (neg_j, neg_l)
        for L in layers:
            tgt[0][L].append(j[L]); tgt[1][L].append(l[L])
        kept["pos" if it.should_fire else "neg"] += 1
    return pos_j, pos_l, neg_j, neg_l, kept, attrition


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    lens_ckpt = sys.argv[2] if len(sys.argv) > 2 else "out/qwen8b_lens150.pt"
    backend = load_qwen_backend(model, lens_ckpt=lens_ckpt)
    backend.enable_thinking = False   # Anchor A is no_cot: NO reasoning text -> C truly silent
    reader, tok = backend.reader, backend.tokenizer
    layers = sorted({int(x) for x in getattr(backend.lens, "source_layers", range(35))})
    items = [it for it in build_counterfactuals() if it.family != "direct_named"]

    probe = ["The Eiffel Tower stands in the middle of the French capital city of",
             "Photosynthesis converts sunlight, water and carbon dioxide into oxygen and",
             "The largest ocean on Earth, covering a third of the surface, is the"]
    l_deg, mean_corr = measure_l_deg(reader, layers, probe)
    print(f"L_deg = {l_deg}  (first layer with J~logit corr >= {L_DEG_CORR}); "
          f"mid = layers < {l_deg}")

    out = {"l_deg": l_deg, "mean_corr": {str(L): round(mean_corr[L], 3) for L in layers}}
    for anchor in ("A", "P"):
        pj, pl, nj, nl, kept, attr = collect(backend, reader, tok, items, layers, anchor)
        print(f"\n=== Anchor {anchor} === kept pos={kept['pos']} neg={kept['neg']}  attrition={attr}")
        if kept["pos"] < 10 or kept["neg"] < 10:
            print("  <10 per side; treat as diagnostic, not a verdict.")
        rows = per_layer_table(pj, pl, nj, nl, layers, l_deg)
        out[f"anchor_{anchor}"] = {"kept": kept, "attrition": attr, "rows": rows}
        print(f"{'layer':>5} {'AUROC_J':>8} {'AUROC_lg':>9} {'dAUROC':>8} {'95%CI':>16} {'perm_p':>7}  band")
        for r in rows:
            print(f"{r['layer']:>5} {r['auroc_j']:>8} {r['auroc_logit']:>9} {r['delta']:>8}"
                  f"  [{r['lo']:>5},{r['hi']:>5}] {r['p']:>7}  {r['band']}")
        if anchor == "A":
            mid_wins = [r for r in rows if r["band"] == "mid" and r["lo"] > 0]
            print(f"\n  VERDICT: {len(mid_wins)} mid-layers (of {sum(1 for r in rows if r['band']=='mid')}) "
                  f"with dAUROC CI-lower > 0 "
                  f"{'-> possible workspace signal, proceed to causal gates' if mid_wins else '-> NO mid-layer Jacobian>logit win: KILL'}")

    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/jacobian_vs_logit_sweep.json", "w"), indent=2)
    print("\nsaved -> out/jacobian_vs_logit_sweep.json")


if __name__ == "__main__":
    main()
