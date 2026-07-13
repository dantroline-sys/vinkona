#!/bin/bash
# Build librnnoise (xiph/rnnoise) — entirely in-tree.
#
# The library is built in var/build/rnnoise and installed to var/rnnoise (no
# sudo, nothing under /usr/local). rnnoise_frontend.py looks there first (.so
# on Linux, .dylib on macOS), so no linker configuration is needed. On
# container setups run INSIDE the distrobox where the cascade executes; on
# macOS or a plain host, run it directly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
BUILD_DIR="${BUILD_DIR:-$VINKONA_VAR/build/rnnoise}"
PREFIX="$VINKONA_VAR/rnnoise"

echo "== Checking build tools =="
if [ "$(uname -s)" = Darwin ]; then
    # clang + make ship with the Xcode Command Line Tools; Homebrew's libtool
    # installs libtoolize as *g*libtoolize (autoreconf honours $LIBTOOLIZE).
    command -v cc >/dev/null 2>&1 \
        || { echo "No C compiler — install the Xcode Command Line Tools first:  xcode-select --install"; exit 1; }
    vk_require_tools git autoconf automake "glibtoolize:libtool" pkg-config || exit 1
    command -v libtoolize >/dev/null 2>&1 || export LIBTOOLIZE=glibtoolize
else
    # autogen.sh needs libtoolize (shipped by the 'libtool' package), NOT the
    # standalone /usr/bin/libtool binary (which is a separate 'libtool-bin'
    # package on Debian/Ubuntu) — so check for libtoolize, which actually runs.
    vk_require_tools git gcc make autoconf automake "libtoolize:libtool" pkg-config || exit 1
fi

echo "== Cloning / updating xiph/rnnoise =="
if [ -d "$BUILD_DIR/.git" ]; then
    git -C "$BUILD_DIR" pull --ff-only
else
    mkdir -p "$(dirname "$BUILD_DIR")"
    git clone https://github.com/xiph/rnnoise.git "$BUILD_DIR"
fi

cd "$BUILD_DIR"
echo "== autogen (downloads the default model) =="
./autogen.sh
echo "== configure + make (prefix: $PREFIX) =="
./configure --prefix="$PREFIX"
make -j"$(vk_ncpu)"

echo "== install librnnoise (in-tree, no sudo) =="
make install

echo "== soxr (Python resampler) — ships with the core set uv installs =="
"$SCRIPT_DIR/vinkona_env/bin/python" -c "import soxr" 2>/dev/null \
    || { echo "ERROR: soxr missing from vinkona_env — run './install.sh core' first."; exit 1; }

echo ""
echo "Done.  librnnoise lives at $PREFIX/lib — verify with:"
echo "  vinkona_env/bin/python -c \"from rnnoise_frontend import RNNoiseFrontend; RNNoiseFrontend(); print('rnnoise + soxr OK')\""
