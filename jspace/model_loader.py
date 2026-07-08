"""Robust HF decoder loading for jlens.

Qwen3.6 checkpoints are tagged `image-text-to-text` (multimodal), so
`AutoModelForCausalLM` may refuse to load them. We fall back to the
image-text-to-text class and extract the text decoder that `jlens.from_hf`
needs (the stack with the unembedding). Verify the extracted module on the box
with `inspect_model(...)` if the fallback path is hit.
"""
from __future__ import annotations

import sys


def _dtype(name: str):
    import torch
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def load_lm(model_id: str, dtype: str = "bf16", device: str = "cuda",
            trust_remote_code: bool = True):
    """Return (model, tokenizer) suitable for `jlens.from_hf`.

    Tries a plain causal LM first; on failure, loads the multimodal wrapper and
    pulls out the text decoder.
    """
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    td = _dtype(dtype)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=td, trust_remote_code=trust_remote_code)
        return model.to(device), tok
    except Exception as e:  # noqa: BLE001
        print(f"[load_lm] AutoModelForCausalLM failed ({type(e).__name__}: {e}); "
              f"trying the image-text-to-text loader", file=sys.stderr)

    Cls = getattr(transformers, "AutoModelForImageTextToText", None) or transformers.AutoModel
    full = Cls.from_pretrained(model_id, torch_dtype=td,
                               trust_remote_code=trust_remote_code).to(device)
    lm = _extract_text_decoder(full)
    print(f"[load_lm] using text decoder: {type(lm).__name__}", file=sys.stderr)
    return lm, tok


def _extract_text_decoder(full):
    """Find the causal-LM submodule (has an output embedding / lm_head)."""
    for attr in ("language_model", "text_model", "model"):
        sub = getattr(full, attr, None)
        if sub is None:
            continue
        has_head = hasattr(sub, "lm_head") or (
            hasattr(sub, "get_output_embeddings") and sub.get_output_embeddings() is not None)
        if has_head:
            return sub
    # No clearly-headed submodule; hand back the wrapper and let jlens try.
    # If jlens.from_hf errors, run inspect_model() and wire the exact path.
    return full


def inspect_model(model_id: str, trust_remote_code: bool = True):
    """Print architecture WITHOUT downloading weights (config + meta-device build).

    Use this to wire the exact text-decoder path for a multimodal checkpoint:
        python -c "from jspace.model_loader import inspect_model; inspect_model('Qwen/Qwen3.6-27B')"
    """
    import torch
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    text_cfg = getattr(cfg, "text_config", cfg)
    print("config:", type(cfg).__name__, "| architectures:", getattr(cfg, "architectures", None))
    for k in ("model_type", "num_hidden_layers", "hidden_size", "vocab_size"):
        print(f"  text.{k}:", getattr(text_cfg, k, None))

    Cls = getattr(transformers, "AutoModelForImageTextToText", None) or transformers.AutoModel
    try:
        with torch.device("meta"):
            model = Cls.from_config(cfg, trust_remote_code=trust_remote_code)
    except Exception as e:  # noqa: BLE001
        print("meta build failed:", e)
        return
    print("top-level children:", [n for n, _ in model.named_children()])
    lm = _extract_text_decoder(model)
    print("extracted decoder:", type(lm).__name__,
          "| has lm_head:", hasattr(lm, "lm_head"))
