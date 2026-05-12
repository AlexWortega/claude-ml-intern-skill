#!/usr/bin/env bash
# One-command installer for the ml-intern Claude Code skill.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/AlexWortega/claude-ml-intern-skill/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/AlexWortega/claude-ml-intern-skill/main/install.sh | TARGET=codex bash
set -euo pipefail

REPO="https://github.com/AlexWortega/claude-ml-intern-skill"
TARGET="${TARGET:-claude}"   # claude | codex | <custom-path>

case "$TARGET" in
  claude) DEST="$HOME/.claude/skills/ml-intern" ;;
  codex)  DEST="$HOME/.codex/prompts/ml-intern" ;;
  /*|./*|../*) DEST="$TARGET" ;;
  *)      DEST="$HOME/.claude/skills/$TARGET"  # fall-through: TARGET treated as a sub-name under ~/.claude/skills
esac

mkdir -p "$(dirname "$DEST")"

if [ -d "$DEST/.git" ]; then
  echo "[ml-intern] already installed at $DEST — pulling latest"
  git -C "$DEST" pull --ff-only
else
  echo "[ml-intern] cloning into $DEST"
  git clone --depth 1 "$REPO" "$DEST"
fi

if [ ! -f "$DEST/.env" ]; then
  cp "$DEST/.env.example" "$DEST/.env"
  echo "[ml-intern] wrote starter $DEST/.env — fill in HF_TOKEN, TG_BOT_TOKEN, SLACK_* as needed"
fi

chmod +x "$DEST/scripts/"*.sh 2>/dev/null || true

cat <<EOF

[ml-intern] installed at: $DEST

Next:
  1. Fill secrets in $DEST/.env (HF_TOKEN minimum; TG/Slack optional)
  2. Open any \`claude\` session (Claude Code) and just describe an ML task,
     or invoke explicitly with /ml-intern <task>.

Examples:
  /ml-intern implement DeepSeek-V3 at ~100M params, train 100 steps on TinyStories
  /ml-intern fine-tune Qwen2.5-0.5B-Instruct on samsum, report ROUGE
  /ml-intern reproduce the LayerNorm paper at GPT-2 scale

Source: $REPO
EOF
