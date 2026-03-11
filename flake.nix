{
  description =
    "Nix dev shell for ocamler-grpo (llama.cpp + OCaml + uv toolchains)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = system == "x86_64-linux";
            cudaCapabilities = [ "12.0" ];
          };
        };

        lib = pkgs.lib;

        llamaCpp = pkgs.llama-cpp;
        llamaCppCuda = pkgs.llama-cpp.override { cudaSupport = true; };

        huggingfaceCli = pkgs.python312Packages.huggingface-hub;

        commonPackages = with pkgs; [
          cmake
          pkg-config
          curl
          openssl
          git
          git-lfs
          jq
          opam
          ocaml
          ocamlPackages.findlib
          uv
          huggingfaceCli
          direnv
          python312  # Provide Python from Nix for compatibility
        ];

        linuxExtras = lib.optionals pkgs.stdenv.isLinux [
          pkgs.curl.dev
          pkgs.util-linux
          pkgs.which
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cuda_cudart
          pkgs.cudaPackages.cuda_nvcc  # Provides libnvptxcompiler.so for PTX JIT compilation
          pkgs.cudaPackages.cuda_nvrtc
          pkgs.cudaPackages.libnvjitlink
          pkgs.cudaPackages.libcublas
          pkgs.stdenv.cc.cc.lib  # Provides libstdc++.so.6 for Python packages with C++ extensions
        ];

        darwinExtras = lib.optionals pkgs.stdenv.isDarwin [ ];

        mkDevShell = { llamaPkg, enableCuda ? false }:
          let
            # Create a wrapper for llama-server (Linux only, with CUDA setup)
            llamaServerWrapper = if pkgs.stdenv.isLinux then
              pkgs.writeShellScriptBin "llama-server" ''
                # Create a temporary directory for our libcuda.so.1 symlink
                CUDA_STUB_DIR=$(mktemp -d)
                trap "rm -rf $CUDA_STUB_DIR" EXIT

                # Symlink only libcuda.so.1 from system
                if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
                  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_STUB_DIR/libcuda.so.1"
                  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_STUB_DIR/libcuda.so"
                fi

                # Only Nix libraries + our isolated libcuda.so.1
                export LD_LIBRARY_PATH="${
                  pkgs.lib.makeLibraryPath [
                    pkgs.cudaPackages.cuda_cudart
                    pkgs.cudaPackages.cuda_nvcc
                    pkgs.cudaPackages.cuda_nvrtc
                    pkgs.cudaPackages.libnvjitlink
                    pkgs.cudaPackages.libcublas
                    pkgs.cudaPackages.cudatoolkit
                  ]
                }:$CUDA_STUB_DIR"

                # Run the actual llama-server from the llamaPkg
                exec ${llamaPkg}/bin/llama-server "$@"
              ''
            else
              # On Darwin, just use llama-server directly without CUDA wrapper
              pkgs.writeShellScriptBin "llama-server" ''
                exec ${llamaPkg}/bin/llama-server "$@"
              '';
          in pkgs.mkShell {
            packages = commonPackages ++ linuxExtras ++ darwinExtras
              ++ [ llamaServerWrapper ];

            shellHook = ''
              # Use Nix's Python to avoid glibc conflicts
              export UV_PYTHON="${pkgs.python312}/bin/python3.12"
              export UV_PYTHON_DOWNLOADS=never

              ${lib.optionalString enableCuda ''
                # Build CUDA only for Nvidia RTX PRO 4500 Blackwell (compute capability 12.0).
                export NIX_CUDA_ARCHITECTURES=120
                export CMAKE_CUDA_ARCHITECTURES=120
                export TORCH_CUDA_ARCH_LIST="12.0"
              ''}

              ${lib.optionalString pkgs.stdenv.isLinux ''
                # Add CUDA and libstdc++ to library path for PyTorch GPU support (Linux only)
                export LD_LIBRARY_PATH="${
                  lib.makeLibraryPath [
                    pkgs.stdenv.cc.cc.lib
                    pkgs.cudaPackages.cuda_cudart
                    pkgs.cudaPackages.cuda_nvcc
                    pkgs.cudaPackages.cuda_nvrtc
                    pkgs.cudaPackages.libnvjitlink
                    pkgs.cudaPackages.cudatoolkit
                    pkgs.cudaPackages.libcublas
                  ]
                }:''${LD_LIBRARY_PATH:-}"
              ''}

              if test -f .envrc; then
                direnv allow >/dev/null 2>&1 || true
                eval "$(direnv hook bash)"
              fi

              # Customize bash prompt to show nix-shell status with username
              export PS1="\[\033[1;35m\]\u\[\033[0m\]@\[\033[1;34m\][nix-shell:\[\033[1;32m\]ocamler-grpo\[\033[1;34m\]]\[\033[0m\] \[\033[1;36m\]\w\[\033[0m\] \$ "

              if [ -z "''${UV_AUTO_SYNC_DISABLED:-}" ]; then
                echo "[nix] Syncing Python dependencies..."
                uv sync --frozen${lib.optionalString enableCuda " --extra cuda"}
              else
                echo "[nix] Skipping automatic uv sync because UV_AUTO_SYNC_DISABLED is set."
              fi
            '';
          };
      in {
        devShells.default = mkDevShell {
          llamaPkg = llamaCpp;
          enableCuda = false;
        };

        devShells.cuda = mkDevShell {
          llamaPkg = llamaCppCuda;
          enableCuda = true;
        };
      });
}
