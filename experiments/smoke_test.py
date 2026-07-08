#!/usr/bin/env python3
"""Cheapest possible toolchain de-risk (run FIRST on the A100).

Two unknowns to kill before spending on the 27B fit:
  1. Does `jlens` fit + apply actually work end-to-end here, and what shape does
     `lens.apply` return? (JlensReader assumes [position, vocab].)
  2. Does the multimodal Qwen loader path work? (Qwen3.6 is image-text-to-text.)

Default: fit a 5-prompt lens on a SMALL text model and confirm a factual token
surfaces in the readout. Costs pennies and ~2 min.

    # 1. toolchain sanity on a small text model:
    python experiments/smoke_test.py --model Qwen/Qwen3-8B

    # 2. inspect the 27B's module tree WITHOUT downloading 54GB of weights:
    python experiments/smoke_test.py --inspect Qwen/Qwen3.6-27B

    # 3. once both pass, do the real fit (see README GPU runbook).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE = "The capital of France is"
EXPECT = "paris"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--inspect", default=None,
                    help="print a model's architecture (no weight download) and exit")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    if args.inspect:
        from jspace.model_loader import inspect_model
        inspect_model(args.inspect)
        return

    import jlens
    from jspace.model_loader import load_lm
    from jspace.lens import JlensReader

    print(f"[1/4] loading {args.model} ({args.dtype}) ...", file=sys.stderr)
    hf, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    model = jlens.from_hf(hf, tok)

    print("[2/4] fitting a 5-prompt lens (sanity, not a real lens) ...", file=sys.stderr)
    os.makedirs("out", exist_ok=True)
    # Prompts must be longer than skip_first tokens; keep them well over ~20 tokens.
    prompts = [
        "The Eiffel Tower is located in the city of Paris, the capital of France, "
        "and it attracts millions of visitors from around the world every single year.",
        "Water boils at one hundred degrees Celsius at sea level, while it freezes "
        "into solid ice at exactly zero degrees on the very same temperature scale.",
        "The mitochondria is often called the powerhouse of the cell because it "
        "produces most of the chemical energy that living organisms need to function.",
        "In the summer of 1969, human beings first walked on the surface of the Moon, "
        "a historic achievement watched live by hundreds of millions of people on Earth.",
        "The quick brown fox jumps over the lazy sleeping dog near the wide river bank "
        "while several small birds sing quietly in the tall green trees nearby at dawn.",
    ]
    lens = jlens.fit(model, prompts=prompts, max_seq_len=64, skip_first=2,
                     checkpoint_path="out/smoke_lens.pt")

    print("[3/4] applying lens across layers on a factual probe ...", file=sys.stderr)
    n_layers = getattr(hf.config, "num_hidden_layers", None) or \
        getattr(getattr(hf.config, "text_config", hf.config), "num_hidden_layers")
    sample_layers = sorted({max(1, n_layers // 4), n_layers // 2, (3 * n_layers) // 4})
    reader = JlensReader(model=model, lens=lens, tokenizer=tok)
    readouts = reader.readouts(PROBE, positions=[-1], layers=sample_layers, topk=args.topk)

    print(f"\n[4/4] top-{args.topk} readouts for {PROBE!r} (expect {EXPECT!r} to appear):")
    hit = False
    for r in readouts:
        top = ", ".join(r.tokens[:8])
        mark = ""
        if EXPECT in r.tokens:
            mark = f"  <-- {EXPECT!r} @ rank {r.tokens.index(EXPECT)}"
            hit = True
        print(f"  layer {r.layer:>2} pos {r.position}: {top}{mark}")

    print("\nRESULT:", "PASS — jlens fit+apply works, decode shape OK." if hit else
          "INCONCLUSIVE — target not in top-k; check apply() return shape / try more layers.")
    print("If the shape looks wrong, inspect one raw apply() call and adjust "
          "JlensReader in jspace/lens.py.")


if __name__ == "__main__":
    main()
