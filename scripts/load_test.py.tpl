"""Minimal verified load test for __REPO__.

Downloads the repo from the HF Hub, instantiates __MODEL_CLASS__ from the bundled
model.py + config.json, loads weights from model.safetensors, runs one forward
pass on a random tensor, and generates a short sample from a fixed prompt.

Run:  python load_test.py
"""
import importlib.util
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO      = "__REPO__"
MODEL_CLS = "__MODEL_CLASS__"
CONFIG_CLS = MODEL_CLS + "Config"
TOKENIZER = "__TOKENIZER__"   # e.g. "gpt2"
PROMPT    = "__PROMPT__"

def _pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free, _ = torch.cuda.mem_get_info()
        return "cuda" if free > 2 * 1024**3 else "cpu"
    except Exception:
        return "cpu"
DEVICE = _pick_device()
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"[1/5] snapshot_download {REPO} ...")
local = Path(snapshot_download(repo_id=REPO))

print(f"[2/5] importing {MODEL_CLS} from local model.py ...")
spec = importlib.util.spec_from_file_location("_mdl", local / "model.py")
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

cfg_dict = json.loads((local / "config.json").read_text())
cfg_dict.pop("_model_class", None)
config = getattr(mod, CONFIG_CLS)(**cfg_dict)
print(f"      config: {config!r}"[:300])

print(f"[3/5] building model + loading safetensors ...")
model = getattr(mod, MODEL_CLS)(config).to(DEVICE).to(DTYPE)
sd = load_file(local / "model.safetensors", device=DEVICE)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing,    f"missing keys: {missing[:5]}"
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
n_params = sum(p.numel() for p in model.parameters())
print(f"      loaded {len(sd)} tensors, total params = {n_params:,} (~{n_params/1e6:.1f}M)")
model.eval()

print(f"[4/5] forward pass ...")
seq = getattr(config, "max_seq_len", getattr(config, "max_position_embeddings", 64))
x   = torch.randint(0, config.vocab_size, (1, min(64, seq)), device=DEVICE)
with torch.no_grad():
    out = model(x)
logits = out[0] if isinstance(out, tuple) else out
assert torch.isfinite(logits).all(), "non-finite logits"
print(f"      logits shape={tuple(logits.shape)} dtype={logits.dtype} finite=True")

print(f"[5/5] generation from {PROMPT!r} ...")
tok = AutoTokenizer.from_pretrained(TOKENIZER)
ids = tok.encode(PROMPT)
x   = torch.tensor([ids], dtype=torch.long, device=DEVICE)
with torch.no_grad():
    for _ in range(60):
        ctx = x[:, -seq:]
        out = model(ctx)
        next_logits = (out[0] if isinstance(out, tuple) else out)[0, -1].float() / 0.8
        v, _ = torch.topk(next_logits, 40)
        next_logits[next_logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(next_logits, -1), 1).item()
        x   = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        if nxt == tok.eos_token_id and x.shape[1] > 30:
            break
print("-" * 60)
print(tok.decode(x[0].tolist()))
print("-" * 60)
print("\nLOAD_TEST: PASS")
