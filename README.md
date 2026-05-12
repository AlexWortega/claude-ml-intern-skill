# claude-ml-intern-skill

> A Claude Code skill that turns the agent into an **autonomous ML intern**: it researches, implements, trains, **self-verifies**, and **publishes** ML code with the Hugging Face ecosystem. Telegram + Slack milestone alerts.

Behavioral port of [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern) as a single drop-in Claude Code skill — no extra daemon, no extra LLM client, no extra auth. The skill reuses the agent loop, tools, and auto-compaction that Claude Code already provides.

---

## What it does

Given a single instruction like *"implement DeepSeek-V3 at ~100M params and train on TinyStories"*, the skill will, with no further input:

1. **Restate** the task and list unknowns → `TASK.md`
2. **Research** the architecture, dataset, and reference implementations using WebFetch + HF Hub API + GitHub code search → `RESEARCH.md` (URLs + section refs)
3. **Plan** files, hyperparameters, dataset slug, success criteria → `PLAN.md`
4. **Implement** `model.py` + `train.py`, run `py_compile` after each file
5. **Smoke-test**: instantiate, print param count, run a forward pass on random tensors before any training
6. **Train** with NaN guard, LR schedule, checkpointing, eval, generation sampling at milestones
7. **Self-verify** → `VERIFY.md` with six independent verdicts (generation sanity, loss-vs-baseline, eval-tracks-train, data consumption, stderr scan, param-count drift). **A low loss number alone never declares success** — the skill exists in part because of a real failure where a model reported `train_loss=0.03 / eval=0.02` but generated word salad (causal mask leak via the CSA Compressor).
8. **Publish to the Hub** → fresh model repo at `<user>/ml-intern-<slug>-<stamp>` with `model.safetensors`, `config.json`, `model.py`, and the full reproducibility bundle. URL written to `PUBLISHED.md`.
9. **Notify** Telegram + Slack at every milestone: `plan_ready`, `code_ready`, `train_started`, `train_done`, `published`, `error`, `blocker`.

All run artifacts live under `~/ml-intern-runs/<slug>/`.

---

## One-command install

### Claude Code

```bash
curl -fsSL https://raw.githubusercontent.com/AlexWortega/claude-ml-intern-skill/main/install.sh | bash
```

This clones the skill into `~/.claude/skills/ml-intern/` and writes a starter `.env`. Open any `claude` session afterwards and type `/ml-intern …` (or just describe an ML task — the skill triggers on phrases like "implement", "train", "fine-tune", "reproduce paper").

Manual equivalent:

```bash
git clone https://github.com/AlexWortega/claude-ml-intern-skill ~/.claude/skills/ml-intern
cp ~/.claude/skills/ml-intern/.env.example ~/.claude/skills/ml-intern/.env
```

### OpenAI Codex CLI (and any other agent that reads markdown instructions)

```bash
curl -fsSL https://raw.githubusercontent.com/AlexWortega/claude-ml-intern-skill/main/install.sh | TARGET=codex bash
```

This clones into `~/.codex/prompts/ml-intern/`. Use the skill by reading `SKILL.md` at session start (e.g. `codex --instructions ~/.codex/prompts/ml-intern/SKILL.md "implement Mamba-2 at 50M, train on enwik8"`).

For any other agent, the contract is just one file: `SKILL.md` is the system prompt; `scripts/notify.sh` and `scripts/hf_push.sh` are referenced from inside it.

---

## Configuration

After install, fill in `~/.claude/skills/ml-intern/.env`:

| key | required | purpose |
|---|---|---|
| `HF_TOKEN` | yes (for publish) | uploads to your HF Hub namespace |
| `HF_USER` | no | override (else taken from `huggingface-cli whoami`) |
| `TG_BOT_TOKEN`, `TG_CHAT_ID` | no | Telegram milestone alerts |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID` | no | Slack milestone alerts (bot token) |
| `SLACK_WEBHOOK_URL` | no | Slack milestone alerts (incoming webhook, used if bot token absent) |
| `GITHUB_TOKEN` | no | enables `gh search code` from inside the skill |

Missing tokens never crash a run — `notify.sh` is a graceful no-op when keys are absent. `hf_push.sh` does fail loudly without `HF_TOKEN` because a non-published run isn't shipped.

---

## Demonstration

End-to-end on a single RTX A6000 (46 GB), no human intervention beyond the initial prompt:

| run | params | init loss | final loss | best eval | peak GPU | wall (training) |
|---|---|---|---|---|---|---|
| DeepSeek-V3 @ 100M, 100 smoke steps | 122.1M | 10.94 | 4.75 | – | 3.1 GB | 24 s |
| DeepSeek-V4 @ 100M, 100 smoke steps | 129.9M | 12.02 | 4.86 | – | 6.4 GB | 35 s |
| DeepSeek-V4 @ 100M, TinyStories full epoch | 129.9M | 10.94 | 1.33 (eval) | 1.33 | 45 GB | ~6 h |

The V4 case is the most informative: skill received only the V4-Pro paper URL, autonomously read the upstream `inference/model.py`, identified the V4 deltas vs V3 (MQA + 512 head_dim, CSA / HCA, mHC hyper-connections, hash routing, sqrtsoftplus gate, swiglu_limit, O grouped low-rank, MTP, attention sink), implemented a 100M down-scale, ran a smoke test, caught a *causal mask leak via the CSA Compressor* during the first full-epoch run, **diagnosed it itself, fixed it, regenerated samples, verified them, and re-ran**.

### Trained model

The TinyStories-trained DeepSeek-V4 100M from the run above:

→ **https://huggingface.co/AlexWortega/ml-intern-v4-100m-tinystories-20260512-1721**

Sample generation from the published checkpoint:

> *"Once upon a time, the little girl called Lucy went to the beach. She loved to play in the sand and splash in the sand. But one day, she got lost in the water. She called out for help, but nobody was around. Suddenly, Lucy saw a big wave coming towards her. She knew she needed to find a way to go back. She ran up to a nearby shore and carefully found her friends. She was so happy to have found her way back home."*

---

## Layout

```
ml-intern/
  SKILL.md                              # frontmatter + behavior prompt
  install.sh                            # one-command installer
  scripts/
    notify.sh                           # TG + Slack POST, env-driven, no-op safe
    hf_search.sh                        # HF Hub search wrapper
    hf_push.sh                          # safetensors conversion + model card + push
  assets/
    deepseek_v3_100m_blueprint.md       # sizing reference for the V3 case
  .env.example
```

---

## How the milestone events work

The skill calls `bash scripts/notify.sh <event> "<message>"` at every stage transition. Event names match upstream `ML_INTERN_SLACK_AUTO_EVENTS` so you can later move to the real `ml-intern` CLI without renaming anything:

- `plan_ready` — `PLAN.md` written
- `code_ready` — `model.py` + `train.py` compile + smoke-test passes
- `train_started` — training loop entered
- `train_done` — final loss + URL
- `published` — `PUBLISHED.md` written with the HF Hub URL
- `error` — fatal (NaN streak, broken VERIFY, upload failure)
- `blocker` — the agent is stuck and is asking for input

Both Telegram and Slack are POSTed in parallel; missing tokens for either channel silently skip that channel.

---

## Why this exists

Manual reimplementations of new papers eat a day each before any signal arrives. The original `huggingface/ml-intern` solves this with its own agentic loop wired into LiteLLM + Anthropic / OpenAI / local backends. This skill ports the *behavior* of that loop into Claude Code so anyone already running `claude` can have the same intern available with no second LLM client and no second auth. The notification, doom-loop guard, context discipline, and milestone-event names are unchanged from upstream.

Beyond the port: this skill adds **mandatory self-verification** (`VERIFY.md`) and **mandatory Hub publishing** because a model in `/tmp/ckpts/` only is not "shipped" and a low loss alone is not "trained".

---

## Credit

Behavior cloned from [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern). All the smart design decisions (HF-first instinct, doom-loop guard, `ML_INTERN_SLACK_AUTO_EVENTS` event names, context discipline) are theirs. This repo is just the Claude Code skill packaging plus the verify-and-publish loop.

License: Apache 2.0.
