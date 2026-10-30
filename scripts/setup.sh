#!/usr/bin/env bash
set -euo pipefail

# MineCombat setup for a Linux machine without conda.
#
# System packages (JDK 8, Xvfb, OpenGL tools) are checked but not installed
# automatically, because they require sudo. The script supports a user-local
# JDK 8 at ~/.local/opt/jdk8 and a local xvfb-run shim.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10 first." >&2
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" != "3.10" ]]; then
  echo "Warning: MineStudio is tested with Python 3.10; found $PY_VERSION." >&2
fi

# Prefer a user-local JDK 8 if one was extracted without sudo.
LOCAL_JDK8="${HOME}/.local/opt/jdk8/usr/lib/jvm/java-8-openjdk-amd64"
if [[ -x "${LOCAL_JDK8}/bin/java" ]]; then
  export JAVA_HOME="${LOCAL_JDK8}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

if command -v java >/dev/null 2>&1; then
  JAVA_VERSION="$(java -version 2>&1 | head -n 1)"
  if [[ "$JAVA_VERSION" != *"1.8"* && "$JAVA_VERSION" != *'"8.'* ]]; then
    echo "Warning: MineStudio expects JDK 8; found: $JAVA_VERSION" >&2
    echo "  Ubuntu: sudo apt-get install -y openjdk-8-jdk" >&2
    echo "  Or see docs/environment.md for a user-local JDK 8." >&2
  fi
else
  echo "Warning: java not found. Install JDK 8 before running the simulator." >&2
fi

if [[ -z "${HF_ENDPOINT:-}" ]]; then
  echo "Hint: if huggingface.co is unreachable, run:"
  echo "  export HF_ENDPOINT=https://hf-mirror.com"
fi

if ! python3 -m venv .venv 2>/dev/null; then
  echo "python3-venv is missing; falling back to virtualenv."
  rm -rf .venv
  python3 -m pip install --user virtualenv
  python3 -m virtualenv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# MineStudio calls xvfb-run even when a working X display exists.
if ! command -v xvfb-run >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
  cp scripts/xvfb-run .venv/bin/xvfb-run
  chmod +x .venv/bin/xvfb-run
  echo "Installed local xvfb-run shim (uses DISPLAY=${DISPLAY})."
fi

# Work around the MalmoEnv server startup race.
python scripts/patch_minestudio.py || true

# Fix the left-click lock (leftClickCounter=10000) in the engine.
python scripts/patch_minestudio_engine.py || true

echo
echo "Environment ready."
echo "  source .venv/bin/activate"
echo "  python scripts/smoke_test.py --config configs/env/zombie_stage1.yaml --steps 100"
