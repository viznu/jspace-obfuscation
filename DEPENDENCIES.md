# Verified dependencies (checked 2026-07-07)

Before writing code we ground-truthed the proposal's load-bearing assumptions.
Results below; **three corrections** materially change the plan.

## Verified ✓

| Thing | Status | Source |
|---|---|---|
| The paper | **Real.** *Verbalizable Representations Form a Global Workspace in Language Models* (Anthropic, 2026) | https://transformer-circuits.pub/2026/workspace/index.html |
| The code | **Real.** `anthropics/jacobian-lens` — reference impl, *not maintained, not accepting contributions* | https://github.com/anthropics/jacobian-lens |
| The model | **Real.** `Qwen/Qwen3.6-27B` (released 21 Apr 2026, ~4.9M downloads) | https://hf.co/Qwen/Qwen3.6-27B |

### Verified `jlens` API (use these signatures — not the proposal's guesses)
```python
import transformers, jlens
hf   = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B").cuda()
tok  = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = jlens.from_hf(hf, tok)

# FIT (no pre-fit lens ships — see correction #2):
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
# ...or LOAD if you ever get one:
lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")

# APPLY:
lens_logits, model_logits, _ = lens.apply(model, text, positions=[-2], layers=[15])

# PARALLELISE fitting over disjoint prompt slices:
lens = jlens.JacobianLens.merge(lens_a, lens_b, ...)
```

## Corrections that change the plan

**1. The paper does NOT use Qwen.** It evaluates **Claude Sonnet 4.5**, corroborated
on **Haiku 4.5 / Opus 4.5 / Opus 4.6** — closed weights, no activation access.
Qwen3.6-27B is the *repo's* example open-weights decoder ("Examples use Qwen;
other HuggingFace decoders adapt cleanly"). ⇒ **Our headline is a
replication-on-open-weights, not a re-run of the paper's exact setup.** State this
explicitly in the writeup; a reviewer will ask whether workspace findings on
Claude transfer to Qwen (that transfer is itself a contribution / threat-to-validity).

**2. No public pre-fit lens exists.** Searched HF models + datasets for a
jlens/workspace lens for Qwen3.6-27B — **none found.** The proposal's §7.4 top
risk resolves to: **we must fit the lens ourselves.** It's on the week-1 critical
path, not optional. (`jlens.fit`, ~afternoon on 27B.)

**3. Fitting corpus is larger than assumed.** README: the paper's lenses use
**1000 sequences × 128 tokens; ~100 prompts is "usable"** — not the "~25 Pile
prompts" the proposal quotes. Plan for ~100 web-text sequences of length 128.

## Still to confirm on the GPU box (cheap, do in week 1)
- Exact return shape / dtype of `lens.apply` (`lens_logits` layout: `[pos, vocab]`
  assumed in `jspace/lens.py::JlensReader` — verify and adjust the decode loop).
- The workspace layer band for Qwen3.6-27B (the "ignition onset"). `jspace/`
  currently assumes `range(10, 22)` as a placeholder; find it empirically
  (logit-lens-interpretability sweep) before locking metrics.
- SAE availability for the migration analysis (§5.5). Neuronpedia listing for
  Qwen3.6-27B was cited in the proposal but not yet confirmed downloadable; a
  linear probe is the fallback.
