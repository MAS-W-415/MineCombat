#!/usr/bin/env bash
# Remove stale MineStudio instance state after a crashed or interrupted run.
set -euo pipefail

MINESTUDIO_DIR="${MINESTUDIO_DIR:-/tmp/MineStudio}"

rm -rf "${MINESTUDIO_DIR}/MP_tmp" "${MINESTUDIO_DIR}/tmp"
echo "Removed runtime state under ${MINESTUDIO_DIR}"
