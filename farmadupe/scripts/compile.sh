#!/bin/sh
# Personal dev build of llama-server (RelWithDebInfo).
#
# Usage:
#   ./farmadupe/scripts/compile.sh           # CPU only
#   ./farmadupe/scripts/compile.sh --cuda    # enable CUDA (GPU offload)
#
# Build dir is "build". Binary lands at build/bin/llama-server.

set -e

CUDA=OFF
for arg in "$@"; do
    case "$arg" in
        --cuda) CUDA=ON ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="build"

# nvcc from a .deb/runfile CUDA install often is not on PATH; CMake's
# enable_language(CUDA) needs it. Add the standard location if it is missing.
if [ "$CUDA" = ON ] && ! command -v nvcc >/dev/null 2>&1; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    else
        echo "nvcc not found; set PATH or CUDACXX to your CUDA toolkit" >&2
        exit 1
    fi
fi

cd "$REPO_ROOT"

cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA="$CUDA"

cmake --build "$BUILD_DIR" --target llama-server -j

# Smoke check: llama-server --help exits non-zero by convention, so don't trust
# its return code -- assert instead that it prints a substantial usage block.
help_chars=$("$BUILD_DIR/bin/llama-server" --help 2>&1 | wc -c)
if [ "$help_chars" -lt 1000 ]; then
    echo "sanity check failed: --help printed only $help_chars chars (<1000)" >&2
    exit 1
fi
echo "sanity check OK: --help printed $help_chars chars"
