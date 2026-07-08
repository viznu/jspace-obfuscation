#!/usr/bin/env python3
"""Look directly at J-space contents: what is the model poised to say, per layer?

No monitoring, no concealment — just read the Jacobian-lens top-k at each layer
for a prompt, optionally beside the plain logit lens, so you can watch concepts
emerge across the network.

    python experiments/inspect_jspace.py "What currency is used in the country whose capital is Rome?"
    python experiments/inspect_jspace.py "<prompt>" --pos -1 --topk 8 --both
    python experiments/inspect_jspace.py "<prompt>" --highlight italy,euro
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lens", default="out/qwen8b_lens.pt")
    ap.add_argument("--pos", type=int, default=-1, help="token position to read (default last)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--both", action="store_true", help="show logit-lens beside J-lens")
    ap.add_argument("--highlight", default="", help="comma-separated tokens to mark with *")
    ap.add_argument("--chat", action="store_true", help="wrap prompt in the chat template")
    args = ap.parse_args()

    import jlens
    from jspace.model_loader import load_lm
    from jspace.lens import JlensReader

    hf, tok = load_lm(args.model, dtype="bf16")
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    reader = JlensReader(model=model, lens=lens, tokenizer=tok)

    text = args.prompt
    if args.chat:
        text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                       tokenize=False, add_generation_prompt=True)

    hi = {h.strip().lower() for h in args.highlight.split(",") if h.strip()}
    layers = sorted({int(x) for x in getattr(lens, "source_layers", range(32))})

    def readout_line(text, positions, use_jacobian):
        # temporary monkey: JlensReader.readouts always uses jacobian; for the logit
        # baseline call apply directly via token path is overkill — reuse readouts by
        # toggling on the lens object is not exposed, so we decode from apply here.
        import torch
        lens_by_layer, _, _ = reader.lens.apply(reader.model, text, layers=layers,
                                                positions=positions, use_jacobian=use_jacobian)
        out = {}
        for L in layers:
            lg = lens_by_layer[L]
            lg = lg if lg.dim() == 2 else lg.unsqueeze(0)
            vals, idx = torch.topk(lg[0].float(), args.topk)
            out[L] = [tok.decode([int(i)]).strip() for i in idx]
        return out

    j = readout_line(text, [args.pos], use_jacobian=True)
    l = readout_line(text, [args.pos], use_jacobian=False) if args.both else None

    def fmt(toks):
        return "  ".join(("*" + t + "*") if t.lower() in hi else t for t in toks)

    print(f"\nprompt: {args.prompt!r}   position: {args.pos}   top-{args.topk}")
    print("=" * 100)
    header = f"{'layer':>5}  {'J-lens (poised to say)':<48}"
    if args.both:
        header += "| logit-lens"
    print(header)
    print("-" * 100)
    for L in layers:
        row = f"{L:>5}  {fmt(j[L]):<48}"
        if args.both:
            row += "| " + fmt(l[L])
        print(row)


if __name__ == "__main__":
    main()
