#!/bin/sh
# Personal dev build of llama-server (CPU only, RelWithDebInfo).
#
# Usage:
#   ./farmadupe/scripts/compile.sh
#
# Build dir is "build". Binary lands at build/bin/llama-server.
# Symlinks compile_commands.json into the repo root for clangd.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="build"

cd "$REPO_ROOT"

cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGGML_NATIVE=ON \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_AVX512_BF16=ON

ln -sf "$BUILD_DIR/compile_commands.json" "$REPO_ROOT/compile_commands.json"

cmake --build "$BUILD_DIR" --target llama-server -j

"$BUILD_DIR/bin/llama-server" --help 2>&1 >/dev/null
