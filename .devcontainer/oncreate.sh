#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER_NAME:-vscode}"

paths=(
  "/mise"
  "/home/vscode/.cache/uv"
  "${containerWorkspaceFolder:-/workspaces/torchfont}/.venv"
  "${containerWorkspaceFolder:-/workspaces/torchfont}/target"
  "${containerWorkspaceFolder:-/workspaces/torchfont}/node_modules"
  "${containerWorkspaceFolder:-/workspaces/torchfont}/data"
)

for path in "${paths[@]}"; do
  if [[ -e "${path}" ]]; then
    sudo chown -R "${USER_NAME}:${USER_NAME}" "${path}"
  fi
done
