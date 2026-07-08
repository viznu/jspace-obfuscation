#!/usr/bin/env python3
"""Find the workspace layer band empirically (GPU box).

The paper describes the workspace as a middle-layer band with a sharp,
ignition-like onset. Our monitors pool the J-lens over that band, so we must
find it for Qwen3.6-27B rather than hard-code the range(10,22) placeholder.

Method: for probe prompts with a known "poised" target token, apply the fitted
lens at EVERY layer at the final position and measure how strongly the target
surfaces (hit@k and mean reciprocal rank). The band = the contiguous run of
layers where target readouts ignite above chance.

    python experiments/find_workspace_band.py --model Qwen/Qwen3.6-27B \
        --lens out/qwen27b_lens.pt --topk 50

Prints a per-layer curve and a suggested `range(onset, plateau_end)` to paste
into experiments/concealment_sweep.py and jspace/lens.py.
"""
from __future__ import annotations

import argparse
import sys

# (prompt, target) — target is the token poised to be produced/used next.
# Mix of direct completions (surface late) and two-hop bridges (should surface
# mid-band even though they are never emitted), which is exactly the workspace signature.
PROBES = [
    ("The currency used in the country whose capital is Rome is the", "euro"),
    ("The capital city of France is", "paris"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The largest planet in the solar system is", "jupiter"),
    ("The author of 'Romeo and Juliet' is William", "shakespeare"),
    ("The chemical symbol for gold is", "au"),
    ("The opposite of hot is", "cold"),
    ("The country whose capital is Tokyo is", "japan"),
]


def sweep(reader, probes, layers, topk):
    """Return [(layer, hit@k, MRR)] over probes, for the lens's fitted layers."""
    layers = list(layers)
    hits = {l: 0 for l in layers}
    rr = {l: 0.0 for l in layers}
    for prompt, target in probes:
        readouts = reader.readouts(prompt, positions=[-1], layers=layers, topk=topk)
        by_layer = {r.layer: r for r in readouts}
        for l in layers:
            toks = by_layer[l].tokens if l in by_layer else []
            if target in toks:
                hits[l] += 1
                rr[l] += 1.0 / (toks.index(target) + 1)
    n = len(probes)
    return [(l, hits[l] / n, rr[l] / n) for l in layers]


def suggest_band(curve, hit_floor=0.4):
    """Longest contiguous run of layers with hit@k >= hit_floor.

    Returns (onset_layer, end_layer_exclusive) -> paste as range(onset, end).
    """
    best = (None, None, 0)
    i = 0
    while i < len(curve):
        if curve[i][1] >= hit_floor:
            j = i
            while j < len(curve) and curve[j][1] >= hit_floor:
                j += 1
            if j - i > best[2]:
                best = (curve[i][0], curve[j - 1][0] + 1, j - i)
            i = j
        else:
            i += 1
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--lens", default="out/qwen27b_lens.pt")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--hit-floor", type=float, default=0.4)
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import jlens
    from jspace.lens import JlensReader
    from jspace.model_loader import load_lm

    hf, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens) if hasattr(jlens.JacobianLens, "load") \
        else jlens.JacobianLens.from_pretrained(args.lens)
    reader = JlensReader(model=model, lens=lens, tokenizer=tok)

    n_layers = getattr(hf.config, "num_hidden_layers", None) or \
        getattr(getattr(hf.config, "text_config", hf.config), "num_hidden_layers")
    # Only sweep the layers the lens was actually fitted for (target layer is excluded).
    avail = sorted({int(x) for x in getattr(lens, "source_layers", range(n_layers))})
    curve = sweep(reader, PROBES, avail, args.topk)

    print(f"\nWorkspace ignition sweep — {args.model}  (topk={args.topk}, probes={len(PROBES)})")
    print("layer  hit@k   MRR")
    for l, hit, rr in curve:
        bar = "#" * int(hit * 30)
        print(f"{l:>3}   {hit:5.2f}  {rr:5.3f}  {bar}")
    onset, end = suggest_band(curve, args.hit_floor)
    print(f"\nSuggested workspace band (hit@k >= {args.hit_floor}): range({onset}, {end})")
    print("Paste into WORKSPACE in experiments/concealment_sweep.py and JlensReader callers.")


if __name__ == "__main__":
    sys.path.insert(0, __import__("os").path.dirname(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__))))
    main()
