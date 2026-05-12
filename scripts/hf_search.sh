#!/usr/bin/env bash
# Convenience HF Hub search.
# Usage: hf_search.sh datasets|models <query> [limit]
# Prints top hits as JSON (id, downloads, likes).

set -eu
kind="${1:-datasets}"
query="${2:?usage: hf_search.sh datasets|models <query> [limit]}"
limit="${3:-5}"

case "$kind" in
  datasets|models) ;;
  *) echo "first arg must be 'datasets' or 'models'" >&2; exit 2 ;;
esac

url="https://huggingface.co/api/${kind}?search=$(printf '%s' "$query" | jq -sRr @uri)&limit=${limit}"
curl -fsS -m 15 "$url" | jq '[.[] | {id, downloads, likes}]'
