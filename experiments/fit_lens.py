#!/usr/bin/env python3
"""Fit a Jacobian lens for an open-weights model (GPU box).

No pre-fit lens ships for Qwen3.6-27B (see DEPENDENCIES.md), so this is week-1
critical path. The paper's lenses use ~1000 seqs x 128 tokens; ~100 is "usable"
(README §9.3). Fitting needs backward passes -> minutes on a small model, an
afternoon on 27B.

    # small model first (debug the whole loop cheaply):
    python experiments/fit_lens.py --model Qwen/Qwen3.6-1.7B \
        --n-prompts 100 --seq-len 128 --out out/qwen1p7b_lens.pt

    # headline:
    python experiments/fit_lens.py --model Qwen/Qwen3.6-27B \
        --n-prompts 100 --seq-len 128 --out out/qwen27b_lens.pt --dtype bf16

Parallelize by fitting disjoint prompt slices on separate GPUs and combining
with JacobianLens.merge (see --shard / --merge).

NOTE: verified jlens.fit signature is fit(model, prompts=..., checkpoint_path=...).
The community note about "skipping the first ~4 high-norm tokens" is passed via
--skip-first and forwarded best-effort; confirm the real kwarg name against the
installed jlens and adjust `_fit_kwargs` below.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sample_webtext(n_prompts: int, seq_len: int, dataset: str, tokenizer,
                   seed: int = 0, scan_cap: int = 100) -> list[str]:
    """Return n_prompts detokenized windows of seq_len tokens from a web-text
    corpus. Defaults to a small fast corpus (NeelNanda/pile-10k) so init never
    hangs; scans at most scan_cap*n_prompts rows before giving up."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train")
    if hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed)
    text_key = None
    prompts: list[str] = []
    max_scan = scan_cap * n_prompts
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        if text_key is None:
            text_key = "text" if "text" in row else next(iter(row))
        toks = tokenizer(row[text_key], truncation=True, max_length=seq_len)["input_ids"]
        if len(toks) < seq_len:
            continue
        prompts.append(tokenizer.decode(toks[:seq_len]))
        if len(prompts) >= n_prompts:
            break
    if len(prompts) < n_prompts:
        raise RuntimeError(f"only gathered {len(prompts)}/{n_prompts} prompts from "
                           f"{dataset} (scanned {min(i+1, max_scan)} rows)")
    return prompts


def _fit_kwargs(skip_first: int) -> dict:
    """Forward skip_first only if the installed jlens.fit accepts it."""
    import jlens
    sig = inspect.signature(jlens.fit)
    kw = {}
    for name in ("skip_first", "skip_first_n", "skip_tokens"):
        if name in sig.parameters:
            kw[name] = skip_first
            break
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--dataset", default="NeelNanda/pile-10k",
                    help="small fast Pile subset; swap for a streaming corpus if desired")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16,
                    help="skip leading high-norm tokens when fitting (jlens default is 16)")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/lens.pt")
    ap.add_argument("--shard", type=int, default=None,
                    help="fit only prompt slice [shard*size:(shard+1)*size] for parallel fits")
    ap.add_argument("--shard-size", type=int, default=None)
    args = ap.parse_args()

    import jlens
    from jspace.model_loader import load_lm

    print(f"loading {args.model} ({args.dtype}) ...", file=sys.stderr)
    hf, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    model = jlens.from_hf(hf, tok)

    print(f"sampling {args.n_prompts} x {args.seq_len}-tok prompts from {args.dataset} ...",
          file=sys.stderr)
    prompts = sample_webtext(args.n_prompts, args.seq_len, args.dataset, tok)

    if args.shard is not None and args.shard_size:
        s = args.shard * args.shard_size
        prompts = prompts[s:s + args.shard_size]
        args.out = args.out.replace(".pt", f".shard{args.shard}.pt")
        print(f"shard {args.shard}: fitting {len(prompts)} prompts -> {args.out}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # jlens.fit's checkpoint_path holds the ROLLING training accumulator
    # (jacobian_sum/n_done) — NOT a loadable lens. Keep it on a .ckpt path and
    # save the finalized JacobianLens (the return value) to args.out.
    ckpt = args.out + ".ckpt"
    lens = jlens.fit(model, prompts=prompts, checkpoint_path=ckpt,
                     **_fit_kwargs(args.skip_first))
    lens.save(args.out)
    print(f"fitted lens saved -> {args.out}  (rolling checkpoint: {ckpt})")
    return lens


if __name__ == "__main__":
    main()
