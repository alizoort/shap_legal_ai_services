#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cleanup() {
  docker compose down -v >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker compose up -d --build

wait_for_url() {
  local url="$1"
  local attempts="${2:-30}"

  for (( attempt=1; attempt<=attempts; attempt+=1 )); do
    if curl -fsS "$url" >/dev/null; then
      return 0
    fi
    sleep 2
  done

  echo "Timed out waiting for $url" >&2
  return 1
}

wait_for_url "http://127.0.0.1:8000/health"
wait_for_url "http://127.0.0.1:8103/health"
wait_for_url "http://127.0.0.1:8000/legal-ai/ping"

curl -fsS "http://127.0.0.1:8000/legal-ai/ping"
