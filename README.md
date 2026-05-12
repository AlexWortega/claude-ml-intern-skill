# ml-intern — Claude Code skill

A behavioral port of [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern) as a Claude Code skill. Drop it in `~/.claude/skills/` and Claude becomes an autonomous ML intern that researches papers, writes training code, runs it, and reports milestones to Telegram + Slack.

## Install

```bash
git clone https://github.com/AlexWortega/claude-ml-intern-skill ~/.claude/skills/ml-intern
cp ~/.claude/skills/ml-intern/.env.example ~/.claude/skills/ml-intern/.env
# fill in TG_BOT_TOKEN / TG_CHAT_ID / SLACK_BOT_TOKEN / SLACK_CHANNEL_ID
```

Then in any `claude` session:

```
/ml-intern implement DeepSeek-V3 architecture at ~100M params, train 100 steps on TinyStories
```

For headless / remote runs:

```bash
claude -p --permission-mode bypassPermissions \
  --add-dir ~/.claude/skills/ml-intern \
  <<<"Activate the ml-intern skill and implement <model> at ~<N>M params, train on <dataset>"
```

## What it does

For any ML task the skill:

1. Restates the task (`TASK.md`)
2. Researches with HF Hub + WebFetch + GitHub code search (`RESEARCH.md`)
3. Plans (`PLAN.md`)
4. Implements `model.py` + `train.py` with smoke test + NaN guard
5. Trains; logs `step=N loss=X` per step to `train.log`
6. Saves `ckpt.pt` and writes `RESULTS.md`
7. Notifies Telegram + Slack at every milestone

All artifacts go under `~/ml-intern-runs/<slug>/`.

## Milestone events

`plan_ready` · `code_ready` · `train_started` · `train_done` · `error` · `blocker` · `approval_required`

Names match upstream `ML_INTERN_SLACK_AUTO_EVENTS` for parity.

`scripts/notify.sh` is a graceful no-op when tokens are missing — the skill always calls it, never gates on token presence.

## Tested end-to-end

Verified on an RTX A6000:

| run | params | init loss | final loss (step 100) | peak GPU |
|---|---|---|---|---|
| DeepSeek-V3 @ 100M | 122.1M | 10.94 | 4.75 | 3.08 GB |
| DeepSeek-V4 @ 100M (from the V4-Pro paper) | 129.9M | 12.02 | 4.86 | 6.37 GB |

The V4 case is interesting: skill received only the paper URL, autonomously fetched the upstream `inference/model.py`, identified the V4 deltas vs V3 (MQA + 512 head_dim, CSA / HCA, mHC hyper-connections, hash routing, sqrtsoftplus gate, swiglu_limit, O grouped low-rank, MTP, attention sink), and implemented a 100M down-scale that trained to a finite descending loss in ~35 s.

## Layout

```
ml-intern/
  SKILL.md                              # frontmatter + behavior prompt
  scripts/
    notify.sh                           # TG + Slack POST, env-driven, no-op safe
    hf_search.sh                        # convenience HF Hub search wrapper
  assets/
    deepseek_v3_100m_blueprint.md       # sizing reference for the V3 case
  .env.example
```

## Credit

Behavior cloned from [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern). All the smart design decisions (HF-first instinct, doom-loop guard, `ML_INTERN_SLACK_AUTO_EVENTS` event names, context discipline) are theirs.
