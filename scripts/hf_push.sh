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
  # fallback to cached cli token
  [ -f "$HOME/.cache/huggingface/token" ] && HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "hf_push: HF_TOKEN not set (env, $env_file, or huggingface-cli login). Aborting." >&2
  exit 3
fi
export HF_TOKEN

if [ -z "${HF_USER:-}" ]; then
  HF_USER=$(HF_TOKEN_FOR_PY="$HF_TOKEN" python3 -c "import os; from huggingface_hub import HfApi; print(HfApi(token=os.environ['HF_TOKEN_FOR_PY']).whoami()['name'])" 2>/dev/null || true)
fi
if [ -z "${HF_USER:-}" ]; then
  echo "hf_push: could not resolve HF user. Set HF_USER or fix HF_TOKEN." >&2
  exit 4
fi

stamp=$(date +%Y%m%d-%H%M)
base_repo="${HF_USER}/ml-intern-${slug}-${stamp}"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

# Build staging dir (model.safetensors, config.json, model.py, reproducibility bundle, README.md).
# Pass shell args into Python via positional argv; use a quoted heredoc so $ and backticks are literal.
ML_RUN_DIR="$run_dir" ML_STAGE="$stage" ML_SLUG="$slug" ML_HF_USER="$HF_USER" \
ML_SCRIPT_DIR="$script_dir" ML_TOKENIZER="${ML_TOKENIZER:-gpt2}" ML_PROMPT="${ML_PROMPT:-Once upon a time,}" \
python3 - <<'PYEOF'
import os, json, glob, dataclasses
from pathlib import Path

run = Path(os.environ["ML_RUN_DIR"])
stage = Path(os.environ["ML_STAGE"])
slug = os.environ["ML_SLUG"]
hf_user = os.environ["ML_HF_USER"]

import torch
try:
    from safetensors.torch import save_file
except Exception as e:
    print(f"hf_push: safetensors not installed: {e}", flush=True)
    raise SystemExit(5)

# pick ckpt: best per RESULTS.md > newest
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
sd = {k: v.contiguous() for k, v in sd.items() if hasattr(v, "shape")}
save_file(sd, str(stage / "model.safetensors"))
print(f"hf_push: wrote model.safetensors ({len(sd)} tensors)", flush=True)

model_py = run / "model.py"
model_class_name = None
if model_py.exists():
    (stage / "model.py").write_text(model_py.read_text())
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_mlintern_model", str(model_py))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg_cls = next((getattr(mod, n) for n in dir(mod) if n.endswith("Config") and dataclasses.is_dataclass(getattr(mod, n))), None)
        if cfg_cls is not None:
            d = dataclasses.asdict(cfg_cls())
            model_class_name = cfg_cls.__name__.replace("Config", "")
            d["_model_class"] = model_class_name
            (stage / "config.json").write_text(json.dumps(d, indent=2, default=str))
            print(f"hf_push: wrote config.json (class={model_class_name})", flush=True)
    except Exception as e:
        print(f"hf_push: could not synthesize config.json: {e}", flush=True)

# reproducibility bundle
for name in ["RESULTS.md", "VERIFY.md", "TASK.md", "PLAN.md", "RESEARCH.md",
             "DEBUG.md", "gen_samples.log", "train.log", "eval.log",
             "train.py", "train_v2.py", "train_full.py", "generate.py"]:
    src = run / name
    if src.exists():
        try:
            (stage / name).write_text(src.read_text(errors="replace"))
        except Exception:
            pass

for tk in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
    src = run / tk
    if src.exists():
        try:
            (stage / tk).write_text(src.read_text(errors="replace"))
        except Exception:
            pass

# generate model card
metrics = results.read_text() if results.exists() else "(training still in progress — see train.log + eval.log)"
gen = ""
gs = run / "gen_samples.log"
if gs.exists():
    for line in gs.read_text().splitlines():
        if line.startswith("OUTPUT:"):
            gen = line[len("OUTPUT:"):].strip().strip("'\"")
            break

card_md = (
    "---\n"
    "library_name: transformers\n"
    "tags:\n"
    "- ml-intern\n"
    "- pretraining\n"
    "license: apache-2.0\n"
    "---\n\n"
    f"# ml-intern run: {slug}\n\n"
    "Trained autonomously by the [ml-intern Claude Code skill](https://github.com/AlexWortega/claude-ml-intern-skill).\n\n"
    "## Sample generation\n\n"
    f"> {gen[:500] if gen else '(no gen sample found)'}\n\n"
    "## Run metrics\n\n"
    f"{metrics}\n\n"
    "## Reproducibility\n\n"
    "The full source (model.py, training script, train.log, eval.log, VERIFY.md, RESEARCH.md, DEBUG.md, gen_samples.log) is bundled in this repo.\n"
)
(stage / "README.md").write_text(card_md)

# Templated load_test.py — substitute REPO, MODEL_CLASS, TOKENIZER, PROMPT
import os as _os
script_dir = Path(_os.environ.get("ML_SCRIPT_DIR", str(Path(__file__).parent if "__file__" in dir() else ".")))
tpl_path = script_dir / "load_test.py.tpl"
if tpl_path.exists() and model_class_name:
    tok_guess = "gpt2"  # safe default; the skill should override via env when known
    prompt_guess = "Once upon a time,"
    body = tpl_path.read_text() \
        .replace("__REPO__", f"{hf_user}/__REPO_FULL__") \
        .replace("__MODEL_CLASS__", model_class_name) \
        .replace("__TOKENIZER__", _os.environ.get("ML_TOKENIZER", tok_guess)) \
        .replace("__PROMPT__", _os.environ.get("ML_PROMPT", prompt_guess))
    # __REPO_FULL__ stays as a placeholder; it's stamped to the real repo name
    # AFTER create_repo succeeds (which may pick a -2 / -3 suffix on collisions).
    (stage / "load_test.py").write_text(body)
    print("hf_push: staged load_test.py", flush=True)

print("hf_push: stage ready", flush=True)
PYEOF

[ $? -eq 0 ] || { echo "hf_push: staging failed" >&2; exit 7; }

# Create repo + (re-stamp load_test.py with the real repo id) + upload.
out=$(ML_BASE="$base_repo" ML_STAGE="$stage" python3 - <<'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

token = os.environ["HF_TOKEN"]
base  = os.environ["ML_BASE"]
stage = Path(os.environ["ML_STAGE"])
api   = HfApi(token=token)
repo_id = base
for suffix in ["", "-2", "-3", "-4", "-5"]:
    cand = base + suffix
    try:
        create_repo(cand, token=token, repo_type="model", private=False, exist_ok=False)
        repo_id = cand
        break
    except Exception as e:
        msg = str(e)
        if "already created this model repo" in msg or "You already created" in msg or "409" in msg:
            continue
        raise

# stamp the real repo id into load_test.py (placeholder __REPO_FULL__)
lt = stage / "load_test.py"
if lt.exists():
    lt.write_text(lt.read_text().replace("__REPO_FULL__", repo_id.split("/", 1)[-1]))

try:
    upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model",
                  commit_message="ml-intern: initial upload", token=token)
except Exception as e:
    print(f"upload_folder failed, retrying with upload_large_folder: {e}", file=sys.stderr, flush=True)
    try:
        api.upload_large_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model")
    except Exception as e2:
        print(f"upload_large_folder also failed: {e2}", file=sys.stderr, flush=True)
        raise
print(repo_id)
PYEOF
)
rc=$?
if [ $rc -ne 0 ] || [ -z "$out" ]; then
  echo "hf_push: upload failed" >&2
  exit 8
fi

repo_id=$(echo "$out" | tail -1)
url="https://huggingface.co/${repo_id}"
echo "$url" > "$run_dir/PUBLISHED.md"
echo "$url"
