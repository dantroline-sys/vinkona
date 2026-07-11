#!/bin/bash
# Build librnnoise (xiph/rnnoise) + install the soxr resampler.
# Run INSIDE the distrobox where the server executes (it needs the C toolchain
# and the vinkona_env venv).  RNNoise is tiny and builds in seconds.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/rnnoise-build}"

echo "== Checking build tools =="
# autogen.sh needs libtoolize (shipped by the 'libtool' package), NOT the
# standalone /usr/bin/libtool binary (which is a separate 'libtool-bin' package
# on Debian/Ubuntu) — so check for libtoolize, which is what actually runs.
for tool in git gcc make autoconf automake libtoolize pkg-config; do
    command -v "$tool" >/dev/null 2>&1 || {
        pkg="$tool"
        [ "$tool" = "libtoolize" ] && pkg="libtool"
        echo "Missing: $tool — install it first (e.g. dnf install $pkg / apt install $pkg)"; exit 1; }
done

echo "== Cloning / updating xiph/rnnoise =="
if [ -d "$BUILD_DIR/.git" ]; then
    git -C "$BUILD_DIR" pull --ff-only
else
    git clone https://github.com/xiph/rnnoise.git "$BUILD_DIR"
fi

cd "$BUILD_DIR"
echo "== autogen (downloads the default model) =="
./autogen.sh
echo "== configure + make =="
./configure
make -j"$(nproc)"

echo "== install librnnoise =="
# Install to /usr/local/lib so ctypes finds it via the default search path.
if [ -w /usr/local/lib ]; then
    make install
else
    sudo make install
fi
# Refresh the linker cache (no-op if ldconfig isn't present).
( ldconfig 2>/dev/null || sudo ldconfig 2>/dev/null ) || true

echo "== install soxr (Python resampler) into the venv =="
source "$SCRIPT_DIR/vinkona_env/bin/activate"
pip install soxr

echo ""
echo "Done.  Verify with:"
echo "  python -c \"import ctypes, soxr; ctypes.CDLL('librnnoise.so'); print('rnnoise + soxr OK')\""
echo ""
echo "Then enable it in serve.sh by adding:  --denoise"
