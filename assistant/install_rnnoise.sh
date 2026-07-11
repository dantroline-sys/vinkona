#!/bin/bash
# Build librnnoise (xiph/rnnoise) + install the soxr resampler — entirely in-tree.
#
# The library is built in var/build/rnnoise and installed to var/rnnoise (no
# sudo, nothing under /usr/local). rnnoise_frontend.py looks there first, so no
# linker configuration is needed. Run INSIDE the distrobox where the cascade
# executes (it needs the C toolchain and the vinkona_env venv).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
BUILD_DIR="${BUILD_DIR:-$VINKONA_VAR/build/rnnoise}"
PREFIX="$VINKONA_VAR/rnnoise"

echo "== Checking build tools =="
# autogen.sh needs libtoolize (shipped by the 'libtool' package), NOT the
# standalone /usr/bin/libtool binary (which is a separate 'libtool-bin' package
# on Debian/Ubuntu) — so check for libtoolize, which is what actually runs.
vk_require_tools git gcc make autoconf automake "libtoolize:libtool" pkg-config || exit 1

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
make -j"$(nproc)"

echo "== install librnnoise (in-tree, no sudo) =="
make install

echo "== install soxr (Python resampler) into the venv =="
source "$SCRIPT_DIR/vinkona_env/bin/activate"
pip install soxr

echo ""
echo "Done.  librnnoise lives at $PREFIX/lib — verify with:"
echo "  vinkona_env/bin/python -c \"from rnnoise_frontend import RNNoiseFrontend; RNNoiseFrontend(); print('rnnoise + soxr OK')\""
