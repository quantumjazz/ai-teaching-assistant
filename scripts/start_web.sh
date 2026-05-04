#!/bin/sh
set -eu

python scripts/bootstrap_index.py
exec gunicorn --workers 1 --bind "${HOST:-0.0.0.0}:${PORT:-8080}" src.app:app
