#!/bin/sh
# Build the driver and run the black-box predictor tests. Expects uv on PATH.
# Any extra args are forwarded to pytest, e.g. ./run.sh -k kimi_k2 -v
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"

cmake -S "$HERE" -B "$HERE/build"
cmake --build "$HERE/build"

cd "$HERE"
uv run pytest "$@"
