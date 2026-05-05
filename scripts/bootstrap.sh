#!/bin/bash
set -eo pipefail

if [ "${EUID}" -eq 0 ]; then
    echo "❌ Error: bootstrap.sh must not be run as root. Run it as the target user."
    exit 1
fi

NIX_PROFILE_DIR="${NIX_PROFILE:-$HOME/.nix-profile}"
NIX_PROFILE_SCRIPT="$NIX_PROFILE_DIR/etc/profile.d/nix.sh"
NIX_PROFILE_EXPORT="export PATH=\"$NIX_PROFILE_DIR/bin:\$PATH\""
NIX_PROFILE_SOURCE="[ -f \"$NIX_PROFILE_SCRIPT\" ] && . \"$NIX_PROFILE_SCRIPT\""

ensure_shell_profile_line() {
    local profile_file="$1"
    local line="$2"

    touch "$profile_file"
    if ! grep -Fqx "$line" "$profile_file"; then
        printf '\n%s\n' "$line" >>"$profile_file"
    fi
}


# Step 1: Install Nix package manager
echo "📦 Installing Nix package manager..."
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install)
echo "✅ Nix installed"

# Step 2: Source Nix profile and persist it for future shells
echo "🔧 Sourcing Nix profile..."
if [ ! -f "$NIX_PROFILE_SCRIPT" ]; then
    echo "❌ Error: Nix profile script not found at $NIX_PROFILE_SCRIPT"
    exit 1
fi

. "$NIX_PROFILE_SCRIPT"
# Nix shells can reset PATH, so persist the profile bin export in bash startup.
export PATH="$NIX_PROFILE_DIR/bin:$PATH"
ensure_shell_profile_line "$HOME/.bashrc" "$NIX_PROFILE_SOURCE"
ensure_shell_profile_line "$HOME/.bashrc" "$NIX_PROFILE_EXPORT"
echo "✅ Nix profile sourced and persisted in $HOME/.bashrc"

# Step 3: Configure Nix experimental features
echo "⚙️  Configuring Nix experimental features..."
sudo mkdir -p /etc/nix
echo "experimental-features = nix-command flakes" | sudo tee /etc/nix/nix.conf >/dev/null
echo "✅ Configuration completed"

# Step 4: Link NVIDIA CUDA and NVML libraries
echo "🔗 Linking NVIDIA CUDA and NVML libraries..."
cd /home/nixer/ocamler-grpo
mkdir -p .cuda-driver
cd .cuda-driver
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so .
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 .
sudo ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 .
export LD_LIBRARY_PATH="/home/nixer/ocamler-grpo/.cuda-driver/:$LD_LIBRARY_PATH"
cd ..
echo "✅ CUDA and NVML libraries linked"

# Step 5: Enter nix development shell with CUDA support and run remaining steps
echo "🔧 Entering nix development environment with CUDA support..."
nix develop .#cuda --command bash -c '
set -eo pipefail

# Verify we are inside nix shell
if [ -z "$IN_NIX_SHELL" ]; then
    echo "❌ Error: Not inside nix shell"
    exit 1
fi
echo "✅ Inside nix shell"

# Step 6: Install Python dependencies with CUDA support
echo "📦 Installing Python dependencies with CUDA support..."
uv sync --extra cuda
echo "✅ Python dependencies installed"

# Step 7: Verify PyTorch CUDA support
echo "🔍 Verifying PyTorch CUDA support..."
uv run python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}\")"
echo "✅ PyTorch verification complete"
'
echo "✅ Bootstrap complete. Start a new shell with nix develop --impure .#cuda"
