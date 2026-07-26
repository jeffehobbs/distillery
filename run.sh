#!/bin/bash
# distillery launcher. Uses the essentia-explorer venv by default — it already has
# the arm64 essentia dev wheel, demucs, and torch/MPS installed (see that
# project's notes: there is no general arm64 essentia wheel on PyPI, only
# specific dev builds, so reusing that venv saves a fight).
set -euo pipefail
cd "$(dirname "$0")"
# Interpreter search order: an explicit DISTILLERY_PYTHON, a local .venv, then a
# sibling essentia-explorer venv (which already has the arm64 essentia wheel).
for candidate in "${DISTILLERY_PYTHON:-}" "./.venv/bin/python" \
                 "$HOME/Scripts/essentia-explorer/.venv/bin/python"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
done
if [ -z "${PYTHON:-}" ]; then
  cat >&2 <<'EOF'
No suitable Python found. Create one:

    python3.12 -m venv .venv
    ./.venv/bin/pip install -r requirements.txt

or set DISTILLERY_PYTHON to an interpreter that has essentia, demucs, torch,
pedalboard, numpy, scipy and soundfile. See the README.
EOF
  exit 1
fi
# unbuffered: otherwise stage progress lags far behind when output is redirected
# to a file, while yt-dlp/ffmpeg (separate processes) write straight through
export PYTHONUNBUFFERED=1
exec "$PYTHON" -m distillery "$@"
