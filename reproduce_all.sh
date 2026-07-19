#!/usr/bin/env bash
set -euo pipefail
python3 scripts/smoke_test.py
python3 scripts/generate_figures.py
