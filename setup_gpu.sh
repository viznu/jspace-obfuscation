#!/usr/bin/env bash
# One-shot setup for the A100 box (Ampere -> standard CUDA 12.x / torch 2.x template).
# Paste-and-go: clones nothing, assumes you scp'd / git-cloned this repo and cd'd in.
set -euo pipefail

# --- persistent HF cache on the big volume so a 54GB download survives restarts ---
if [ -d /workspace ] && [ -w /workspace ]; then
  export HF_HOME=/workspace/hf
else
  export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

python -m pip install -q --upgrade pip

# torch ships in the template; install everything else. transformers must be new
# enough to know the qwen3_5 / qwen3_5_moe arch -> take latest, use trust_remote_code.
python -m pip install -q --upgrade "transformers" accelerate datasets peft sentencepiece
python -m pip install -q "git+https://github.com/anthropics/jacobian-lens"

echo "--- environment check ---"
python - <<'PY'
import torch
print("torch      :", torch.__version__, "| cuda", torch.version.cuda)
print("device     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("vram (GB)  :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1) if torch.cuda.is_available() else "-")
x = torch.randn(1024, 1024, device="cuda"); print("cuda matmul:", float((x@x).sum()) is not None)
import transformers; print("transformers:", transformers.__version__)
try:
    import jlens
    api = [a for a in dir(jlens) if not a.startswith("_")]
    print("jlens OK   :", ", ".join(sorted(api)[:10]))
except Exception as e:
    print("jlens FAIL :", type(e).__name__, e)
PY

cat <<'EOF'

Setup done. Recommended order:
  1. python experiments/smoke_test.py --model Qwen/Qwen3-8B            # ~2 min, pennies
  2. python experiments/smoke_test.py --inspect Qwen/Qwen3.6-27B       # no weight download
  3. python experiments/fit_lens.py --model Qwen/Qwen3.6-27B --n-prompts 100 \
         --seq-len 128 --dtype bf16 --out out/qwen27b_lens.pt          # the real fit
  4. python experiments/find_workspace_band.py --model Qwen/Qwen3.6-27B --lens out/qwen27b_lens.pt
  5. python experiments/day1_rung0.py --backend qwen --model Qwen/Qwen3.6-27B \
         --lens-ckpt out/qwen27b_lens.pt --workspace-start <onset> --workspace-end <end>

Persist HF_HOME across shells:  echo 'export HF_HOME=$HF_HOME' >> ~/.bashrc
EOF
