#!/bin/sh
# Personal dev build of llama-server (CPU only, RelWithDebInfo).
#
# Usage:
#   ./farmadupe/scripts/compile.sh
#
# Build dir is "build". Binary lands at build/bin/llama-server.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="build"

cd "$REPO_ROOT"

cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_SHARED_LIBS=OFF

cmake --build "$BUILD_DIR" --target llama-server -j

"$BUILD_DIR/bin/llama-server" --help 2>&1 >/dev/null
