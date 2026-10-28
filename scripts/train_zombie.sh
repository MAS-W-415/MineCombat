#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"
python -m minecombat.train.ppo --config configs/ppo/zombie_vpt1x.yaml "$@"
