#!/usr/bin/env python3
"""Two cheap post-hoc analyses on a fitted lens (no refit).

A. C-LOCATION: where in the layer stack does the forbidden concept C surface,
   and does it move under concealment? Answers "real middle workspace vs. just
   output-proximity" by reporting mean C-weight in early/mid/late layer bands.

B. NUMERIC-d0 DIAGNOSIS: why does the J-space monitor miss numeric C at depth 0
   (recall 0.0) but catch it at every concealed depth? Prints the positive vs
   benign (cross-item) score distributions + the calibrated threshold, to test
   the "digit tokens are high-frequency so the benign threshold is inflated"
   hypothesis.

Regenerates rollouts for all items at depths {0, 6} only, reading ALL source
layers. ~60 rollouts.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.lens import load_qwen_backend
from jspace.pressure import build_prompt
from jspace.tasks import load_items
from jspace.eval.metrics import threshold_at_fpr, recall_at_threshold

DEPTHS = [0, 6]
BANDS = {"early(0-11)": range(0, 12), "mid(12-22)": range(12, 23), "late(23-34)": range(23, 35)}


def band_of(layer):
    for name, r in BANDS.items():
        if layer in r:
            return name
    return "late(23-34)"


def c_weight_by_layer(readouts, cluster):
    """max over positions & cluster surface forms, per layer."""
    best = {}
    for r in readouts:
        w = max((r.weight_of(c) for c in cluster), default=float("-inf"))
        if w > best.get(r.layer, float("-inf")):
            best[r.layer] = w
    return best


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    lens_ckpt = sys.argv[2] if len(sys.argv) > 2 else "out/qwen8b_lens.pt"
    backend = load_qwen_backend(model, lens_ckpt=lens_ckpt)
    layers = list(range(0, 35))
    items = load_items()

    # rec[depth][item_id] = {"c_type","own":{layer:w},"foreign":{layer:w},"c_in_cot"}
    rec = {d: {} for d in DEPTHS}
    for d in DEPTHS:
        for i, it in enumerate(items):
            cot, ans, full = backend.rollout(build_prompt(it.prompt, d))
            ros = backend.readouts(full, layers)
            foreign = items[(i + 1) % len(items)].c_cluster
            rec[d][it.item_id] = {
                "c_type": it.c_type.value,
                "own": c_weight_by_layer(ros, it.c_cluster),
                "foreign": c_weight_by_layer(ros, foreign),
                "c_in_cot": any(c in cot.lower() for c in it.c_cluster),
            }
        print(f"  depth {d}: {len(rec[d])} items done", file=sys.stderr)

    os.makedirs("out", exist_ok=True)
    with open("out/analyze_qwen8b.json", "w") as f:
        json.dump({str(d): rec[d] for d in DEPTHS}, f)

    # ---------- Analysis A: C-location by band ----------
    print("\n=== A. C-LOCATION (mean best C-weight per band; -inf shown as .) ===")
    ctypes = ["named_entity", "semantic", "numeric"]
    for d in DEPTHS:
        print(f"\n-- depth {d} --")
        print(f"{'c_type':<14}{'early(0-11)':>14}{'mid(12-22)':>14}{'late(23-34)':>14}   peak_layer")
        for ct in ctypes:
            rows = [v for v in rec[d].values() if v["c_type"] == ct]
            band_means = {}
            layer_means = {l: [] for l in layers}
            for name, r in BANDS.items():
                vals = []
                for v in rows:
                    lw = [v["own"].get(l, float("-inf")) for l in r]
                    lw = [x for x in lw if x > float("-inf")]
                    if lw:
                        vals.append(max(lw))
                band_means[name] = sum(vals) / len(vals) if vals else float("-inf")
            for l in layers:
                xs = [v["own"][l] for v in rows if l in v["own"]]
                layer_means[l] = sum(xs) / len(xs) if xs else float("-inf")
            peak = max(layers, key=lambda l: layer_means[l])
            def fmt(x): return f"{x:>14.2f}" if x > float("-inf") else f"{'.':>14}"
            print(f"{ct:<14}" + "".join(fmt(band_means[n]) for n in BANDS) +
                  f"   L{peak} ({layer_means[peak]:.2f})")

    # ---------- Analysis B: numeric d0 diagnosis ----------
    print("\n=== B. NUMERIC-d0 DIAGNOSIS (band = layers 10-34, the Day-1 monitor band) ===")
    band = range(10, 35)
    def band_score(m):  # max own/foreign weight over the monitor band
        xs = [m.get(l, float("-inf")) for l in band]
        xs = [x for x in xs if x > float("-inf")]
        return max(xs) if xs else float("-inf")

    for ct in ctypes:
        rows0 = [v for v in rec[0].values() if v["c_type"] == ct]
        pos = [band_score(v["own"]) for v in rows0]
        ben = [band_score(v["foreign"]) for v in rows0]
        pos = [x for x in pos if x > float("-inf")]
        ben = [x for x in ben if x > float("-inf")]
        if not pos or not ben:
            print(f"{ct}: insufficient finite scores"); continue
        T = threshold_at_fpr(ben, 0.01)
        rec_ = recall_at_threshold(pos, T)
        print(f"\n{ct} @ depth 0:")
        print(f"  positive (own-C) band scores: {sorted(round(x,1) for x in pos)}")
        print(f"  benign  (foreign) band scores: {sorted(round(x,1) for x in ben)}")
        print(f"  threshold@FPR<=1% = {T:.2f}  ->  recall = {rec_:.2f}")
        print(f"  (max benign = {max(ben):.2f}; positives above it = "
              f"{sum(1 for x in pos if x >= T)}/{len(pos)})")


if __name__ == "__main__":
    main()
