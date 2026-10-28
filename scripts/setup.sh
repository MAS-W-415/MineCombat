#!/usr/bin/env bash
set -euo pipefail

# MineCombat setup for a Linux machine without conda.
#
# The script creates a local virtual environment and installs the package.
# System packages (JDK 8, Xvfb, OpenGL tools) are checked but not installed
# automatically, because they require sudo.

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

if command -v java >/dev/null 2>&1; then
  JAVA_VERSION="$(java -version 2>&1 | head -n 1)"
  if [[ "$JAVA_VERSION" != *"1.8"* && "$JAVA_VERSION" != *'"8.'* ]]; then
    echo "Warning: MineStudio expects JDK 8; found: $JAVA_VERSION" >&2
    echo "  Ubuntu: sudo apt-get install -y openjdk-8-jdk" >&2
  fi
else
  echo "Warning: java not found. Install JDK 8 before running the simulator." >&2
fi

if ! command -v Xvfb >/dev/null 2>&1 && [[ -z "${DISPLAY:-}" ]]; then
  echo "Warning: no Xvfb and no DISPLAY. Install Xvfb or use a desktop session." >&2
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo
echo "Environment ready."
echo "  source .venv/bin/activate"
echo "  python scripts/smoke_test.py --config configs/env/zombie_stage1.yaml --steps 100"
