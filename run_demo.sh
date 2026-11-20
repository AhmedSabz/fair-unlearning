#!/usr/bin/env bash
# One-command live demo:  ./run_demo.sh      (then the browser opens http://localhost:8257)
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null; then PY=python3
else PY=python; fi
exec "$PY" demo/app.py "$@"
