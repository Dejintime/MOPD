#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${MOPD_PYTHON:-/home/yangdejin/miniconda3/envs/mopd/bin/python}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
exec "$PYTHON_BIN" -u -m bandit_mopd.train "$@"
