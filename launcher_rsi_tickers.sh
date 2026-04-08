#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"


source "$SCRIPT_DIR/venv/bin/activate"

python "$SCRIPT_DIR/tickets.py"
