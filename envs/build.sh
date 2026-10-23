#!/bin/sh
# Build one isolated venv per system from envs/<name>.txt, then record the exact
# resolved set in envs/<name>.lock. Usage: envs/build.sh NAME [NAME...]
set -e
cd "$(dirname "$0")/.."
for name in "$@"; do
  uv venv -q --allow-existing -p python3.12 ".venvs/$name"
  uv pip install -q -p ".venvs/$name" -r "envs/$name.txt"
  uv pip freeze -p ".venvs/$name" > "envs/$name.lock"
  echo "built $name"
done
