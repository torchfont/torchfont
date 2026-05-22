#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER_NAME:-vscode}"

paths=(
  "/mise"
  "/home/vscode/.cache/uv"
  ".venv"
  "target"
  "node_modules"
  "data"
)

for path in "${paths[@]}"; do
  if [[ -e "${path}" ]]; then
    sudo chown -R "${USER_NAME}:${USER_NAME}" "${path}"
  fi
done
