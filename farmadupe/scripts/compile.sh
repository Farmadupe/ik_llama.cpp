#!/bin/sh
# Personal dev build of llama-server (RelWithDebInfo).
#
# Usage: 
#   ./farmadupe/scripts/compile.sh [--cpu]
# Flags:
#   --cpu: CPU-only build

set -e

CUDA=ON
for arg in "$@"; do
    case "$arg" in
        --cpu) CUDA=OFF ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$REPO_ROOT"

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CUDA_HOST_COMPILER=clang++ \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA="$CUDA"

cmake --build build --target llama-server -j

# Smoke check: llama-server --help exits non-zero by convention, so don't trust
# its return code -- assert instead that it prints a substantial usage block.
help_chars=$("build/bin/llama-server" --help 2>&1 | wc -c)
if [ "$help_chars" -lt 1000 ]; then
    echo "sanity check failed: --help printed only $help_chars chars (<1000)" >&2
    exit 1
fi
echo "sanity check OK: --help printed $help_chars chars"
