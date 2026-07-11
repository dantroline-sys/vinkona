#!/bin/bash
# =============================================================================
# LEGACY: PersonaPlex-mode installer (the original speech-to-speech path).
# The current cascade stack installs with ./install.sh — see README "Legacy" section.
# Supports: Fedora (bare metal) and Ubuntu (including cuda distrobox)
#
# Recommended Ubuntu container: cuda:12.6.3-devel-ubuntu24.04
#   distrobox create --image nvidia/cuda:12.6.3-devel-ubuntu24.04 --name vinkona-cuda
#   distrobox enter vinkona-cuda
#
# Rust mode replaces personaplex_server.py entirely — the moshi-backend binary
# handles WebRTC signaling, audio pipeline, and CUDA inference natively.
# Python mode uses personaplex_server.py and is simpler to debug.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
MODEL_DIR="$SCRIPT_DIR/Models/personaplex-7b-v1-raw"
HF_REPO="nvidia/personaplex-7b-v1"
MOSHI_SRC="$SCRIPT_DIR/moshi_src"
OS=""

# ── Terminal colours ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
die()     { echo -e "${RED}[FATAL]${RESET} $*" >&2; exit 1; }

# ── OS detection ─────────────────────────────────────────────────────────────
detect_os() {
    if grep -qi "fedora" /etc/os-release 2>/dev/null; then
        OS="fedora"
    elif grep -qi "ubuntu\|debian" /etc/os-release 2>/dev/null; then
        OS="ubuntu"
    else
        die "Unsupported OS. This script supports Fedora and Ubuntu/Debian."
    fi
    info "Detected OS: $OS"
}

# Unified package install — callers use package names for the detected OS.
pkg_install() {
    case "$OS" in
        fedora) sudo dnf install -y "$@" ;;
        ubuntu) sudo apt-get install -y "$@" ;;
    esac
}

# ── Environment checks ────────────────────────────────────────────────────────
check_root() {
    if [[ "$EUID" -eq 0 ]]; then
        die "Do not run as root. Use a regular user with sudo access."
    fi
}

check_nvidia() {
    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi not found. Ensure NVIDIA drivers are installed (or that " \
            "GPU access is configured for this container)."
    fi
    info "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    info "GPU(s) found:  $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' '|' | sed 's/|$//')"
}

# ── CUDA toolkit ──────────────────────────────────────────────────────────────
# In the cuda:12.6.3-devel-ubuntu24.04 distrobox image nvcc is pre-installed.
# On bare-metal Fedora/Ubuntu it may need to be installed. Either way this
# function exports all four env vars that cudarc's build.rs looks for so the
# Rust build never fails with "Failed to execute nvcc: No such file or directory".
ensure_cuda_toolkit() {
    local nvcc_path=""
    for candidate in \
            "$(command -v nvcc 2>/dev/null)" \
            /usr/local/cuda/bin/nvcc \
            /usr/bin/nvcc; do
        if [[ -x "$candidate" ]]; then
            nvcc_path="$candidate"
            break
        fi
    done

    if [[ -z "$nvcc_path" ]]; then
        info "nvcc not found — installing CUDA toolkit ..."
        case "$OS" in
            fedora)
                local repo_added=0
                for fver in 40 39; do
                    local url="https://developer.download.nvidia.com/compute/cuda/repos/fedora${fver}/x86_64/cuda-fedora${fver}.repo"
                    if sudo dnf config-manager --add-repo "$url" 2>/dev/null; then
                        repo_added=1; break
                    fi
                done
                [[ "$repo_added" -eq 1 ]] || die "Could not add NVIDIA CUDA repo for Fedora. Add it manually and re-run."
                sudo dnf install -y cuda-toolkit
                ;;
            ubuntu)
                # For bare Ubuntu without the devel image, pull from NVIDIA's apt repo.
                # If you are already inside the cuda devel distrobox this branch is never reached.
                sudo apt-get install -y wget gnupg
                wget -qO /tmp/cuda-keyring.deb \
                    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
                sudo dpkg -i /tmp/cuda-keyring.deb
                sudo apt-get update -qq
                sudo apt-get install -y cuda-toolkit-12-6
                ;;
        esac
        nvcc_path=/usr/local/cuda/bin/nvcc
        [[ -x "$nvcc_path" ]] || \
            die "Toolkit install finished but nvcc still missing at $nvcc_path. Check the log above."
        success "CUDA toolkit installed."
    else
        info "nvcc found: $nvcc_path"
    fi

    # Derive CUDA root from wherever nvcc actually lives (handles non-standard paths).
    local cuda_bin_dir cuda_root
    cuda_bin_dir="$(dirname "$nvcc_path")"
    cuda_root="$(dirname "$cuda_bin_dir")"

    export PATH="${cuda_bin_dir}${PATH:+:$PATH}"
    export CUDA_PATH="$cuda_root"
    export CUDA_ROOT="$cuda_root"
    export CUDA_TOOLKIT_ROOT_DIR="$cuda_root"
    export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    local ver
    ver=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    success "CUDA toolkit ready: $ver  (root: $cuda_root)"
    echo "$ver"
}

detect_cuda_version() {
    local ver="unknown"
    if command -v nvcc &>/dev/null; then
        ver=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    elif [[ -f /usr/local/cuda/version.json ]]; then
        ver=$(grep -oP '"cuda" : "\K[0-9]+\.[0-9]+' /usr/local/cuda/version.json 2>/dev/null || echo "unknown")
    elif [[ -f /usr/local/cuda/version.txt ]]; then
        ver=$(grep -oP "CUDA Version \K[0-9]+\.[0-9]+" /usr/local/cuda/version.txt 2>/dev/null || echo "unknown")
    fi
    echo "$ver"
}

# Maps CUDA X.Y to the nearest PyTorch wheel tag available on download.pytorch.org.
cuda_to_torch_tag() {
    local ver="$1"
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if   [[ "$major" -ge 13 ]];                            then echo "cu128"
    elif [[ "$major" -eq 12 && "$minor" -ge 8 ]];          then echo "cu128"
    elif [[ "$major" -eq 12 && "$minor" -ge 4 ]];          then echo "cu124"  # covers 12.4–12.7 (incl. 12.6)
    elif [[ "$major" -eq 12 && "$minor" -ge 1 ]];          then echo "cu121"
    else
        warn "CUDA $ver is older than 12.1 — defaulting to cu121 (may not work)"
        echo "cu121"
    fi
}

# ── Model download ────────────────────────────────────────────────────────────
download_model() {
    info "Checking for model weights in $MODEL_DIR ..."
    if [[ -d "$MODEL_DIR" && "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]]; then
        success "Model directory already populated — skipping download."
        return
    fi
    mkdir -p "$MODEL_DIR"

    if ! command -v huggingface-cli &>/dev/null; then
        info "Installing huggingface_hub CLI ..."
        pip install --quiet huggingface_hub
    fi

    if ! huggingface-cli whoami &>/dev/null; then
        echo ""
        warn "Not logged into HuggingFace. Run:  huggingface-cli login"
        warn "Then re-run this installer."
        exit 1
    fi

    info "Downloading $HF_REPO → $MODEL_DIR"
    info "(~14 GB at fp16 — this will take a while)"
    huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR"
    success "Model downloaded."
}

# ── Python installation ───────────────────────────────────────────────────────
install_python() {
    info "=== Python + CUDA mode ==="

    case "$OS" in
        fedora)
            if ! command -v python3.12 &>/dev/null; then
                info "Installing python3.12 via dnf ..."
                sudo dnf install -y python3.12 python3.12-devel python3.12-pip
            fi
            ;;
        ubuntu)
            info "Ensuring python3.12, venv, and dev headers are present ..."
            sudo apt-get update -qq
            sudo apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip
            ;;
    esac

    info "Creating virtualenv: vinkona_env"
    python3.12 -m venv "$SCRIPT_DIR/vinkona_env"
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/vinkona_env/bin/activate"
    pip install --upgrade --quiet pip wheel

    local cuda_ver
    cuda_ver=$(detect_cuda_version)
    if [[ "$cuda_ver" == "unknown" ]]; then
        warn "Could not detect CUDA version — defaulting to cu124."
        cuda_ver="12.4"
    fi
    local torch_tag
    torch_tag=$(cuda_to_torch_tag "$cuda_ver")
    info "CUDA $cuda_ver → PyTorch wheel: $torch_tag"

    info "Installing PyTorch (CUDA build) ..."
    pip install --quiet torch --index-url "https://download.pytorch.org/whl/${torch_tag}"

    info "Verifying CUDA is visible to PyTorch ..."
    python3 - <<'PYCHECK'
import torch, sys
avail = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"  CUDA available: {avail}  |  Devices: {count}")
if not avail:
    print("ERROR: PyTorch cannot see a CUDA device. Check driver and toolkit.")
    sys.exit(1)
PYCHECK

    info "Installing remaining Python dependencies (core + legacy PersonaPlex stack) ..."
    pip install --quiet -r "$SCRIPT_DIR/requirements.txt" \
                        -r "$SCRIPT_DIR/requirements-personaplex.txt" \
                        --extra-index-url "https://download.pytorch.org/whl/${torch_tag}"

    success "Python environment ready."
    download_model
    generate_python_serve_script
}

generate_python_serve_script() {
    local out="$SCRIPT_DIR/serve.sh"
    info "Writing $out ..."
    cat > "$out" <<'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vinkona_env/bin/activate"

# 0 = first GPU (RTX 3090), 1 = second GPU (RTX 4090).
# Use the higher-VRAM card for best inference throughput.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$SCRIPT_DIR/personaplex_server.py" \
  --hf-repo nvidia/personaplex-7b-v1 \
  --lm-config         "$SCRIPT_DIR/Models/personaplex-7b-v1-raw/config.json" \
  --moshi-weight      "$SCRIPT_DIR/Models/personaplex-7b-v1-raw/model.safetensors" \
  --text-tokenizer    "$SCRIPT_DIR/Models/personaplex-7b-v1-raw/tokenizer_spm_32k_3.model" \
  --voice-prompt-dir  "$SCRIPT_DIR/Models/personaplex-7b-v1-raw/voices" \
  -q 8 \
  --debug
EOF
    chmod +x "$out"
    success "serve.sh written."
}

# ── Rust installation ─────────────────────────────────────────────────────────
install_rust() {
    info "=== Rust + CUDA mode ==="
    info "Builds kyutai moshi-backend from source. Replaces personaplex_server.py."
    info "Estimated build time: 5–15 min on a fast CPU."

    info "Installing system build dependencies ..."
    case "$OS" in
        fedora)
            sudo dnf install -y openssl-devel pkg-config alsa-lib-devel clang git cmake
            ;;
        ubuntu)
            sudo apt-get update -qq
            # libclang-dev provides clang-sys headers required by some crates.
            # libasound2-dev provides ALSA for audio I/O.
            sudo apt-get install -y \
                libssl-dev \
                pkg-config \
                libasound2-dev \
                libclang-dev \
                clang \
                git \
                cmake \
                build-essential
            ;;
    esac

    # cudarc (Rust CUDA bindings) requires nvcc at build time.
    # This is always pre-installed in the cuda:12.6.3-devel-ubuntu24.04 image.
    local cuda_ver
    cuda_ver=$(ensure_cuda_toolkit)

    # Rust toolchain
    if ! command -v cargo &>/dev/null; then
        info "Rust toolchain not found — installing via rustup ..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
        rustup default stable
    else
        info "Rust found: $(rustc --version)"
        rustup update stable
    fi
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"

    # Clone Kyutai moshi repo (shallow — we only need the tip)
    if [[ -d "$MOSHI_SRC" ]]; then
        info "Updating existing source in $MOSHI_SRC ..."
        git -C "$MOSHI_SRC" pull --ff-only
    else
        info "Cloning kyutai-labs/moshi ..."
        git clone --depth 1 https://github.com/kyutai-labs/moshi "$MOSHI_SRC"
    fi

    # Locate the Rust backend crate (handles both single-crate and workspace layouts)
    local rust_backend_dir="$MOSHI_SRC/rust/moshi-backend"
    [[ ! -d "$rust_backend_dir" ]] && rust_backend_dir="$MOSHI_SRC/rust"
    [[ -f "$rust_backend_dir/Cargo.toml" ]] || \
        die "Cannot find Cargo.toml under $MOSHI_SRC/rust. Check the repo layout."

    info "Building with --features cuda ..."
    cargo build \
        --release \
        --features cuda \
        --manifest-path "$rust_backend_dir/Cargo.toml"

    # Find the compiled binary.
    # Cargo workspaces put all binaries under the workspace root's target/, not
    # the individual crate's directory — so we search broadly rather than assuming.
    local binary=""
    while IFS= read -r candidate; do
        if [[ -x "$candidate" ]]; then
            binary="$candidate"
            break
        fi
    done < <(find "$MOSHI_SRC/rust/target/release" -maxdepth 1 -type f -name "moshi*" 2>/dev/null | sort)

    if [[ -z "$binary" ]]; then
        echo ""
        echo -e "${RED}[FATAL]${RESET} Build finished but no moshi* binary found." >&2
        echo    "        Searched: $MOSHI_SRC/rust/target/release" >&2
        echo    "        Contents of that directory:" >&2
        ls -lh "$MOSHI_SRC/rust/target/release" 2>/dev/null | grep -v '\.d$' | head -20 >&2
        echo    "        Run:  find ~/vinkona/moshi_src/rust/target -name 'moshi*' -type f" >&2
        exit 1
    fi

    info "Copying $binary → $SCRIPT_DIR/moshi-backend"
    cp "$binary" "$SCRIPT_DIR/moshi-backend"
    success "Rust binary ready."

    # Minimal pip install just to get huggingface-cli for the model download
    pip install --quiet --user huggingface_hub 2>/dev/null || \
        python3 -m pip install --quiet --user huggingface_hub 2>/dev/null || true
    download_model

    generate_rust_serve_script
    print_rust_config_note
}

generate_rust_serve_script() {
    local out="$SCRIPT_DIR/serve_rust.sh"
    info "Writing $out ..."
    cat > "$out" <<'RUSTSERVE'
#!/bin/bash
# Rust moshi-backend launcher.
# Run:  ./moshi-backend --help   to verify CLI flags and adjust below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# For a 3090+4090 rig, prefer the 4090 (index 1) for best inference throughput.
export CUDA_VISIBLE_DEVICES=1
export RUST_LOG=info

"$SCRIPT_DIR/moshi-backend" \
  --config    "$SCRIPT_DIR/Models/personaplex-7b-v1-raw/config.json" \
  --model-dir "$SCRIPT_DIR/Models/personaplex-7b-v1-raw" \
  --port 8001 \
  "$@"
RUSTSERVE
    chmod +x "$out"
    success "serve_rust.sh written."
}

print_rust_config_note() {
    echo ""
    echo -e "${BOLD}${YELLOW}=== Rust mode: next steps ===${RESET}"
    echo "  1. Run:  ./moshi-backend --help"
    echo "     Verify the CLI flags — update serve_rust.sh if they differ from above."
    echo "  2. The Rust binary handles WebRTC directly."
    echo "     Point your Flutter client at port 8001 (or whatever you configure)."
    echo "  3. personaplex_server.py is NOT used in Rust mode."
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────
print_banner() {
    echo ""
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║   Vinkona PersonaPlex Installer — CUDA Edition       ║${RESET}"
    echo -e "${BOLD}${CYAN}║   Fedora (bare metal) · Ubuntu · CUDA distrobox    ║${RESET}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "  ${BOLD}1)${RESET} Python  —  moshi + FastAPI/aiortc + personaplex_server.py"
    echo -e "             Simpler to debug. ~15–30 ms overhead per audio cycle."
    echo ""
    echo -e "  ${BOLD}2)${RESET} Rust    —  moshi-backend binary (replaces Python server entirely)"
    echo -e "             Best latency. ~1–3 ms overhead. ~5–15 min build time."
    echo ""
}

main() {
    check_root
    detect_os
    check_nvidia
    print_banner

    local mode
    if [[ $# -gt 0 ]]; then
        mode="$1"
        info "Non-interactive mode: option $mode"
    else
        read -rp "Select install mode [1/2]: " mode
    fi

    case "$mode" in
        1) install_python ;;
        2) install_rust   ;;
        *) die "Invalid selection '$mode'. Choose 1 (Python) or 2 (Rust)." ;;
    esac

    echo ""
    success "══════════════════════════════════════════"
    success "Installation complete."
    if [[ "$mode" == "1" ]]; then
        success "Launch with:  ./serve.sh"
    else
        success "Launch with:  ./serve_rust.sh"
        success "Check flags:  ./moshi-backend --help"
    fi
    success "══════════════════════════════════════════"
}

main "$@"
