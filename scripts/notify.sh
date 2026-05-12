#!/usr/bin/env bash
# ml-intern skill notifier — Telegram + Slack
# Usage: notify.sh <event> "<message>"
# Graceful no-op if tokens unset. Always exits 0 so the skill never crashes on a missing webhook.

set -u
event="${1:-event}"
message="${2:-}"

# Source .env from skill folder if present (next to this script's parent).
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$(dirname "$script_dir")/.env"
if [ -f "$env_file" ]; then
  # shellcheck disable=SC1090
  set -a; . "$env_file"; set +a
fi

host="$(hostname -s 2>/dev/null || echo unknown)"
text="[ml-intern@${host}] ${event}: ${message}"

# --- Telegram ---
if [ -n "${TG_BOT_TOKEN:-}" ] && [ -n "${TG_CHAT_ID:-}" ]; then
  if ! curl -fsS -m 10 \
        "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TG_CHAT_ID}" \
        --data-urlencode "text=${text}" \
        >/dev/null 2>&1; then
    echo "notify.sh: telegram send failed" >&2
  fi
else
  echo "notify.sh: TG_BOT_TOKEN/TG_CHAT_ID unset — skipping telegram" >&2
fi

# --- Slack ---
if [ -n "${SLACK_BOT_TOKEN:-}" ] && [ -n "${SLACK_CHANNEL_ID:-}" ]; then
  payload=$(jq -nc --arg c "$SLACK_CHANNEL_ID" --arg t "$text" '{channel:$c, text:$t}')
  if ! curl -fsS -m 10 \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -H "Content-type: application/json; charset=utf-8" \
        -d "$payload" \
        https://slack.com/api/chat.postMessage \
        >/dev/null 2>&1; then
    echo "notify.sh: slack send failed" >&2
  fi
elif [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  payload=$(jq -nc --arg t "$text" '{text:$t}')
  if ! curl -fsS -m 10 \
        -H "Content-type: application/json" \
        -d "$payload" \
        "$SLACK_WEBHOOK_URL" \
        >/dev/null 2>&1; then
    echo "notify.sh: slack webhook send failed" >&2
  fi
else
  echo "notify.sh: SLACK_BOT_TOKEN/SLACK_CHANNEL_ID and SLACK_WEBHOOK_URL unset — skipping slack" >&2
fi

exit 0
