{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    rustc
    cargo
    rust-analyzer
    pkg-config
    openssl
    maturin
    python3
  ];

  shellHook = ''
    echo "Kryonix Brain Rust Dev Shell"
    export RUST_BACKTRACE=1
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.python3 pkgs.stdenv.cc.cc.lib ]}:$LD_LIBRARY_PATH"
  '';
}
