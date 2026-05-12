#!/usr/bin/env bash
# Push an ml-intern run to the HF Hub as a model repo.
# Usage: hf_push.sh <run-dir> <slug>
#
# Requires HF_TOKEN in env or .env (sibling of this script's parent dir).
# Optional: HF_USER override (else huggingface-cli whoami).

set -u
run_dir="${1:?usage: hf_push.sh <run-dir> <slug>}"
slug="${2:?usage: hf_push.sh <run-dir> <slug>}"

if [ ! -d "$run_dir" ]; then
  echo "hf_push: run dir not found: $run_dir" >&2
  exit 2
fi

# source skill .env if present
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$(dirname "$script_dir")/.env"
[ -f "$env_file" ] && { set -a; . "$env_file"; set +a; }

if [ -z "${HF_TOKEN:-}" ]; then
  echo "hf_push: HF_TOKEN not set (env or $env_file). Aborting." >&2
  exit 3
fi
export HF_TOKEN

# resolve user
if [ -z "${HF_USER:-}" ]; then
  HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi(token='${HF_TOKEN}').whoami()['name'])" 2>/dev/null || true)
fi
if [ -z "${HF_USER:-}" ]; then
  echo "hf_push: could not resolve HF user. Set HF_USER or fix HF_TOKEN." >&2
  exit 4
fi

stamp=$(date +%Y%m%d-%H%M)
base_repo="${HF_USER}/ml-intern-${slug}-${stamp}"

# stage files in a temp dir
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

# Convert best (or final) ckpt -> safetensors
python3 - <<PY
import os, json, glob, dataclasses
from pathlib import Path
import torch
try:
    from safetensors.torch import save_file
except Exception as e:
    print(f"hf_push: safetensors not installed: {e}", flush=True)
    raise SystemExit(5)

run = Path("$run_dir")
stage = Path("$stage")

# pick ckpt: best > step_final > newest
ckpts = sorted(glob.glob(str(run / "ckpts" / "*.pt")))
if not ckpts:
    print("hf_push: no ckpts/*.pt found", flush=True); raise SystemExit(6)
best = None
results = run / "RESULTS.md"
if results.exists():
    for line in results.read_text().splitlines():
        if "ckpt_best_path" in line:
            cand = line.split("|")[-2].strip()
            if cand and (run / cand).exists():
                best = str(run / cand); break
if best is None:
    best = ckpts[-1]
print(f"hf_push: using ckpt {best}", flush=True)

state = torch.load(best, map_location="cpu", weights_only=False)
sd = state.get("model", state) if isinstance(state, dict) else state
# strip any non-tensor keys
sd = {k: v.contiguous() for k, v in sd.items() if hasattr(v, "shape")}
save_file(sd, str(stage / "model.safetensors"))
print(f"hf_push: wrote model.safetensors ({len(sd)} tensors)", flush=True)

# config.json — try to find <Model>Config in model.py and serialize
model_py = run / "model.py"
if model_py.exists():
    (stage / "model.py").write_text(model_py.read_text())
    cfg_path = stage / "config.json"
    try:
        import sys, importlib.util
        spec = importlib.util.spec_from_file_location("_mlintern_model", str(model_py))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg_cls = next((getattr(mod, n) for n in dir(mod) if n.endswith("Config") and dataclasses.is_dataclass(getattr(mod, n))), None)
        if cfg_cls is not None:
            d = dataclasses.asdict(cfg_cls())
            model_cls = next((n for n in dir(mod) if n != cfg_cls.__name__ and n.replace("Config","") in cfg_cls.__name__.replace("Config","")), None) or cfg_cls.__name__.replace("Config","")
            d["_model_class"] = model_cls
            cfg_path.write_text(json.dumps(d, indent=2, default=str))
            print(f"hf_push: wrote config.json (class={model_cls})", flush=True)
    except Exception as e:
        print(f"hf_push: could not synthesize config.json: {e}", flush=True)

# copy reproducibility bundle
for name in ["RESULTS.md", "VERIFY.md", "TASK.md", "PLAN.md", "RESEARCH.md",
             "DEBUG.md", "gen_samples.log", "train.log", "eval.log",
             "train.py", "train_v2.py", "train_full.py", "generate.py"]:
    src = run / name
    if src.exists():
        (stage / name).write_text(src.read_text(errors="replace"))

# tokenizer if present
for tk in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
    src = run / tk
    if src.exists():
        (stage / tk).write_text(src.read_text(errors="replace"))

# model card
card = stage / "README.md"
if not card.exists():
    metrics = ""
    if results.exists():
        metrics = results.read_text()
    gen = ""
    gs = run / "gen_samples.log"
    if gs.exists():
        for line in gs.read_text().splitlines():
            if line.startswith("OUTPUT:"):
                gen = line[len("OUTPUT:"):].strip().strip("'\"")
                break
    card.write_text(f"""---
library_name: transformers
tags:
- ml-intern
- pretraining
license: apache-2.0
---

# ml-intern run: {slug}

Trained autonomously by [ml-intern](https://github.com/AlexWortega/claude-ml-intern-skill).

## Sample generation

> {gen[:500] if gen else "(no gen sample found)"}

## Run metrics

{metrics if metrics else "(see RESULTS.md)"}

## Reproducibility

`model.py`, training scripts, full `train.log`, `VERIFY.md` are bundled in this repo.
""")
print("hf_push: stage ready", flush=True)
PY

[ $? -eq 0 ] || { echo "hf_push: staging failed" >&2; exit 7; }

# create + upload via huggingface_hub
out=$(python3 - <<PY
from huggingface_hub import HfApi, create_repo
api = HfApi(token="$HF_TOKEN")
base = "$base_repo"
for suffix in ["", "-2", "-3", "-4", "-5"]:
    repo_id = base + suffix
    try:
        create_repo(repo_id, token="$HF_TOKEN", repo_type="model", private=False, exist_ok=False)
        break
    except Exception as e:
        if "already created" in str(e) or "You already created" in str(e):
            continue
        # else re-raise
        raise
print(repo_id)
try:
    api.upload_folder(folder_path="$stage", repo_id=repo_id, repo_type="model", commit_message="ml-intern: initial upload")
except Exception:
    api.upload_large_folder(folder_path="$stage", repo_id=repo_id, repo_type="model")
PY
) || { echo "hf_push: upload failed" >&2; exit 8; }

repo_id=$(echo "$out" | tail -1)
url="https://huggingface.co/${repo_id}"
echo "$url" > "$run_dir/PUBLISHED.md"
echo "$url"
