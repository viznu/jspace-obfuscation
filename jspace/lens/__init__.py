"""Thin wrapper over `anthropics/jacobian-lens` (verified API), plus an offline
MOCK backend so the whole Day-1 pipeline runs and the metrics are validated
without a GPU.

Verified API (github.com/anthropics/jacobian-lens, reference impl):
    model = jlens.from_hf(hf_model, tokenizer)
    lens  = jlens.fit(model, prompts=..., checkpoint_path=...)           # fitting
    lens  = jlens.JacobianLens.from_pretrained(repo, filename="model/lens.pt")
    lens_logits, model_logits, _ = lens.apply(model, text, positions=[...], layers=[...])
    JacobianLens.merge(...)   # combine lenses fit on disjoint prompt slices

NOTE (verified 2026-07-07): the paper's own lenses are for Claude; NO public
pre-fit lens for Qwen3.6-27B was found on HF. The real path is `jlens.fit` on
~100 web-text prompts of length 128. See DEPENDENCIES.md.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from ..types import Readout


class RolloutModel(Protocol):
    """Generates a (cot, answer) rollout for a prompt."""
    def rollout(self, prompt: str) -> tuple[str, str]: ...


class LensReader(Protocol):
    """Returns J-lens top-k readouts for `text` at the requested layers/positions."""
    def readouts(self, text: str, positions: Sequence[int], layers: Sequence[int],
                 topk: int = 50) -> list[Readout]: ...


# ----------------------------- real backend -----------------------------------

@dataclass
class JlensReader:
    """Wraps a fitted jlens.JacobianLens + a jlens-wrapped HF model.

    Constructed by `load_qwen_backend(...)`; kept import-guarded so the analysis
    spine imports fine on a laptop without torch/jlens installed.
    """
    model: object          # jlens-wrapped HF model
    lens: object           # jlens.JacobianLens
    tokenizer: object

    def readouts(self, text: str, positions: Sequence[int], layers: Sequence[int],
                 topk: int = 50) -> list[Readout]:
        import torch  # local import: only needed on the GPU box

        # Verified jlens.apply return: (dict{layer: [n_pos, vocab]}, model_logits, targets).
        # One call handles all layers; index the dict per layer.
        lens_by_layer, _model_logits, _targets = self.lens.apply(
            self.model, text, layers=list(layers), positions=list(positions)
        )
        out: list[Readout] = []
        for layer in layers:
            logits = lens_by_layer[layer]            # [n_positions, vocab]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            for pi, pos in enumerate(positions):
                vals, idx = torch.topk(logits[pi].float(), k=topk)
                toks = [self.tokenizer.decode([int(i)]).strip().lower() for i in idx]
                out.append(Readout(layer=int(layer), position=int(pos),
                                   tokens=toks, weights=[float(v) for v in vals]))
        return out

    def token_logits(self, text: str, positions: Sequence[int], layers: Sequence[int],
                     token_ids: Sequence[int], use_jacobian: bool = True) -> dict:
        """Direct lens-logit of specific vocab token ids, regardless of top-k rank.

        Returns {layer: [max-over-token_ids logit at each position]}. This is the
        decision signal the top-k membership rule misses: a concept present in the
        J-space but ranked below k still has a real (often high) logit here.

        `use_jacobian=False` gives the plain logit-lens readout (decode the
        residual directly, no Jacobian transport) — the baseline the ablation
        compares against, through the same code path.
        """
        import torch

        lens_by_layer, _m, _t = self.lens.apply(
            self.model, text, layers=list(layers), positions=list(positions),
            use_jacobian=use_jacobian)
        tid = torch.tensor(sorted(set(int(t) for t in token_ids)))
        out = {}
        for layer in layers:
            logits = lens_by_layer[layer]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            vals = logits[:, tid].float().max(dim=1).values  # [n_positions]
            out[int(layer)] = [float(v) for v in vals]
        return out

    def token_logprobs(self, text: str, positions: Sequence[int], layers: Sequence[int],
                       token_ids: Sequence[int], use_jacobian: bool = True,
                       reduce: str = "mean") -> dict:
        """Log-prob (log-softmax) of C's token ids per layer/position, reduced over
        the cluster. Log-prob removes each lens's temperature/offset so the Jacobian
        and logit arms are comparable; mean-over-cluster avoids the max-of-k bias
        toward the higher-variance lens. `use_jacobian=False` = plain logit lens.
        """
        import torch

        lens_by_layer, _m, _t = self.lens.apply(
            self.model, text, layers=list(layers), positions=list(positions),
            use_jacobian=use_jacobian)
        tid = torch.tensor(sorted(set(int(t) for t in token_ids)))
        out = {}
        for layer in layers:
            logits = lens_by_layer[layer]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            logits = logits.float()
            logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # [n_pos, vocab]
            sel = logp[:, tid]                                             # [n_pos, k]
            red = sel.mean(dim=1) if reduce == "mean" else sel.max(dim=1).values
            out[int(layer)] = [float(v) for v in red]
        return out


@dataclass
class QwenBackend:
    """Real GPU backend: generates rollouts AND reads the J-lens, sharing one
    loaded model. `--backend qwen` in the Day-1 driver uses this.
    """
    hf: object
    tokenizer: object
    model: object          # jlens-wrapped
    lens: object
    device: str = "cuda"
    max_new_tokens: int = 512
    read_positions: int = 16      # read the lens at the last K positions of the sequence
    enable_thinking: bool = True  # use the model's NATIVE chain-of-thought (<think>)

    @property
    def reader(self) -> JlensReader:
        return JlensReader(model=self.model, lens=self.lens, tokenizer=self.tokenizer)

    def rollout(self, prompt: str) -> tuple[str, str, str]:
        """Greedy-decode a rollout; return (cot, answer, full_text).

        `full_text` (prompt + generation) is what the lens reads over. The CoT is
        the model's native <think> block when `enable_thinking` (the faithful
        setting for a study about monitoring chain-of-thought), plus any
        <reasoning> block; the answer is the <answer> tag or the text after
        </think>.
        """
        import torch

        msgs = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking)
        except TypeError:  # model/template without the kwarg
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.hf.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        # Keep special tokens so <think>/</think> survive; strip only chat controls.
        gen = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=False)
        for ctrl in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            gen = gen.replace(ctrl, "")

        think = _between(gen, "<think>", "</think>")
        reasoning = _between(gen, "<reasoning>", "</reasoning>")
        cot = "\n".join(x for x in (think, reasoning) if x) or gen.strip()

        answer = _between(gen, "<answer>", "</answer>")
        if not answer:
            tail = gen.split("</think>", 1)[1] if "</think>" in gen else gen
            tail = tail.strip()
            answer = tail if tail else gen.strip()
        return cot, answer, text + gen.strip()

    def readouts(self, full_text: str, layers) -> list[Readout]:
        positions = list(range(-self.read_positions, 0))
        return self.reader.readouts(full_text, positions=positions, layers=list(layers))


def load_qwen_backend(model_id: str = "Qwen/Qwen3.6-27B",
                      lens_ckpt: Optional[str] = None,
                      fit_prompts: Optional[Sequence[str]] = None,
                      device: str = "cuda",
                      dtype: str = "bf16") -> "QwenBackend":
    """Load model + fit-or-load lens. Runs on the GPU box, not the laptop.

    If `lens_ckpt` exists it is loaded; else a lens is fitted from `fit_prompts`
    (~100 web-text sequences of length 128) and saved to `lens_ckpt`.
    """
    import jlens
    from ..model_loader import load_lm

    hf, tok = load_lm(model_id, dtype=dtype, device=device)
    model = jlens.from_hf(hf, tok)
    lens = _fit_or_load_lens(jlens, model, lens_ckpt, fit_prompts)
    return QwenBackend(hf=hf, tokenizer=tok, model=model, lens=lens, device=device)


def _fit_or_load_lens(jlens, model, lens_ckpt, fit_prompts):
    if lens_ckpt and _exists(lens_ckpt):
        if hasattr(jlens.JacobianLens, "load"):
            return jlens.JacobianLens.load(lens_ckpt)
        return jlens.JacobianLens.from_pretrained(lens_ckpt)
    if not fit_prompts:
        raise ValueError("no lens checkpoint found and no fit_prompts provided; "
                         "run experiments/fit_lens.py first")
    return jlens.fit(model, prompts=list(fit_prompts), checkpoint_path=lens_ckpt)


def _between(s: str, start: str, end: str) -> str:
    i = s.find(start)
    if i == -1:
        return ""
    j = s.find(end, i + len(start))
    return s[i + len(start): j].strip() if j != -1 else s[i + len(start):].strip()


# ------------------------------ mock backend ----------------------------------
# Offline simulation to validate plumbing + metrics. Emits readouts whose
# C-cluster weight follows a chosen hypothesis so we can confirm the metric code
# discriminates H1/H2/H3 BEFORE spending GPU time. These numbers are SYNTHETIC.

@dataclass
class MockLens:
    """Deterministic mock of a workspace band of readouts.

    hypothesis:
      "H1" -> J-space robust: C stays strong in readouts at every cue depth.
      "H2" -> co-collapse: C weight in readouts decays with cue depth.
      "H3" -> concept-dependent: numeric C decays with depth; entity/semantic stay.
    """
    workspace_layers: range = field(default_factory=lambda: range(10, 22))
    positions: tuple[int, ...] = (-3, -2, -1)
    hypothesis: str = "H3"

    def readouts_for_item(self, c_cluster: Sequence[str], c_type: str,
                          cue_depth: int, is_positive: bool) -> list[Readout]:
        out: list[Readout] = []
        base = 8.0 if is_positive else -6.0  # benign traces: C not in workspace
        decay = self._decay(c_type, cue_depth)
        c_weight = base - (decay if is_positive else 0.0)
        for layer in self.workspace_layers:
            # ignition-like: weight peaks mid-band, tapers at edges
            center = (self.workspace_layers.start + self.workspace_layers.stop) / 2
            span = max(1.0, (self.workspace_layers.stop - self.workspace_layers.start) / 2)
            layer_gain = 1.0 - min(1.0, abs(layer - center) / span) * 0.4
            for pos in self.positions:
                w = c_weight * layer_gain + self._jitter(c_cluster, layer, pos)
                filler = self._filler(layer, pos)
                tokens = [c_cluster[0]] + [t for t, _ in filler]
                weights = [w] + [wt for _, wt in filler]
                out.append(Readout(int(layer), int(pos), tokens, weights))
        return out

    def _decay(self, c_type: str, depth: int) -> float:
        if self.hypothesis == "H1":
            return 0.0
        if self.hypothesis == "H2":
            return 2.4 * depth
        # H3: numeric flees the workspace; lexical/semantic stay put
        return 2.6 * depth if c_type == "numeric" else 0.3 * depth

    @staticmethod
    def _jitter(c_cluster, layer, pos) -> float:
        h = hashlib.md5(f"{c_cluster[0]}|{layer}|{pos}".encode()).hexdigest()
        return (int(h[:4], 16) / 0xFFFF - 0.5) * 1.5

    @staticmethod
    def _filler(layer, pos):
        toks = ["the", "a", "of", "and", "to", "is", "in", "that", "it", "answer"]
        out = []
        for i, t in enumerate(toks):
            h = hashlib.md5(f"{t}|{layer}|{pos}|{i}".encode()).hexdigest()
            out.append((t, (int(h[:4], 16) / 0xFFFF) * 3.0))
        return out


def _exists(path: str) -> bool:
    import os
    return os.path.exists(path)
