# Results

## Summary

This project asked whether an **activation-level monitor** — a readout of a model's
internal state via the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html)
("J-space") — can catch a model reasoning about something it does not say, in cases where
a **text (chain-of-thought) monitor** would miss it.

After a pre-registered stress test, the answer on the model studied here (Qwen3-8B) is
**negative**: the Jacobian lens provides **no advantage over a plain logit lens** for this
purpose. It does not read the concealed intermediate concept in the middle layers of the
network any better than the simplest possible baseline; where the concept does surface,
the logit lens is actually *better*. Earlier positive-looking results were artifacts of
**where** the readout was taken (late layers, and over the model's own generated text),
not evidence of a special internal "workspace" channel.

This is a deliberately narrow, honest negative result. It does not refute the original
workspace paper (whose findings are on Claude, a different and closed model); it stress-
tests one specific safety-motivated *use* of that idea on one open model and finds it does
not hold up.

## The decisive test

The plain logit lens already reads a model's internal state by projecting it through the
output vocabulary. Near the final layer, the Jacobian lens is mathematically almost the
same thing — so for the Jacobian lens to be worth anything as a distinct monitor, it must
beat the logit lens in the **middle** layers, where the logit lens is known to be
uninformative. That single comparison is the make-or-break test.

**Setup.** A lens was fit properly (112 web-text prompts, past the point where quality
saturates). Tasks are two-hop questions whose answer requires a hidden bridge entity — e.g.
*"What currency is used in the country whose capital is Rome?"* requires computing *Italy*
to reach *Euro*. To read a genuinely **silent** representation, the model is given no room
to reason and the readout is taken at the moment it commits to the answer, at a position
where the bridge entity appears nowhere in the text. Positives (the bridge is the required
route) are compared against answer-matched and prompt-matched controls (same answer via a
different country; the entity named in the prompt but a different route). Detection is
measured per layer as AUROC, with paired bootstrap confidence intervals and a permutation
test. The "degeneracy layer" (where the two lenses become the same) is measured, not
assumed, so "middle" is defined honestly.

**Result (silent answer-commit read, 17 positive / 18 control traces):**

| layers | finding |
|---|---|
| 0–22 | Both lenses at chance (AUROC ≈ 0.5). Neither reads the bridge in early/middle layers. |
| 23–26 | The concept surfaces — but the **logit lens beats the Jacobian lens, significantly**: e.g. layer 24, logit 0.95 vs Jacobian 0.77 (Δ = −0.18, p = 0.008); layer 25, 0.98 vs 0.78 (p = 0.013). |
| 27–34 | Both saturate near 1.0. |

**There is no middle layer where the Jacobian lens beats the logit lens.** The pre-
registered kill criterion — a band of middle layers with a Jacobian advantage whose
confidence interval excludes zero — is not met. The test fails.

## Why an earlier version looked positive

An earlier run reported the Jacobian monitor "holding" under concealment where the text
monitor collapsed. Two confounds explain it, both now ruled out:

1. **Reading position.** That run read the lens over the model's *generated* tokens — where
   the model was literally writing the entity. At a genuinely silent position, the
   advantage disappears.
2. **Late layers.** The signal lives near the output, where *any* readout (including the
   trivial logit lens) sees the concept because the model is about to produce it. That is
   not the "middle-layer workspace" the claim requires.

## One real but insufficient signal

When the readout is taken at the *question-reading* position (not the silent answer-commit
position), the properly-fit Jacobian lens does show a modest middle-layer edge over the
logit lens. This is a genuine property of the lens — it can surface an entailed concept
mid-network that a logit lens cannot. But that position reflects the model *openly
comprehending* the question ("the country whose capital is Rome" ⟹ Italy), which is not
concealed reasoning, so it does not support the safety-monitoring claim. It was pre-
registered as a diagnostic, not a headline.

## Scope and caveats

- One model (Qwen3-8B), one concept type (named-entity / country), prompt-based concealment,
  small samples. The negative is specific to this setting.
- Concealment here is *instructed*, not a behavior the model chose — a weaker pressure than
  the concealment the safety literature worries about.
- The workspace paper's own results are on Claude; this says nothing about that model.

## What was not built, and why

The plan had further stages — a learned "tuned lens" baseline, a supervised probe as an
honest ceiling, causal interventions, and a train-against-the-monitor attack. All of them
were gated on this first test passing. Because it did not, they are moot and were not built.
The pre-registered kill criterion did its job: it ended the line of work cheaply, before
those stages were built on a premise that does not hold.

Raw outputs from the run are in `out/qwen8b_run/` — the per-layer table is in
`jacobian_vs_logit_sweep.json`.
