#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

LOCAL_JDK8="${HOME}/.local/opt/jdk8/usr/lib/jvm/java-8-openjdk-amd64"
if [[ -x "${LOCAL_JDK8}/bin/java" ]]; then
  export JAVA_HOME="${JAVA_HOME:-${LOCAL_JDK8}}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

export RAY_AUTH_MODE="${RAY_AUTH_MODE:-disabled}"

if ray status >/dev/null 2>&1; then
  echo "Ray cluster already running."
  ray status
  exit 0
fi

ray start --head \
  --resources='{"database": 10000, "rollout_workers": 100, "wandb": 1}' \
  --port 9899 \
  --disable-usage-stats

sleep 3
ray status
