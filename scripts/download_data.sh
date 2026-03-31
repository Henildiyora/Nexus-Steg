#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_DIR="$ROOT_DIR/datasets/DIV2K_train_HR"
VALID_DIR="$ROOT_DIR/datasets/DIV2K_valid_HR"

DIV2K_TRAIN_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
fail()  { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*"; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "'$1' is required but not installed."
}

count_images() {
    find "$1" -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l
}

# ── DIV2K download ───────────────────────────────────────────────────────────

download_div2k_split() {
    local name="$1" url="$2" dest_dir="$3"

    info "Checking $name images in $dest_dir ..."

    if [ -d "$dest_dir" ]; then
        existing=$(count_images "$dest_dir")
        if [ "$existing" -ge 10 ]; then
            ok "$name already has $existing images — skipping download."
            return
        fi
    fi

    need_cmd wget
    need_cmd unzip

    local zip_name
    zip_name="$(basename "$url")"
    local zip_path="$ROOT_DIR/datasets/$zip_name"

    mkdir -p "$ROOT_DIR/datasets"

    info "Downloading DIV2K $name (~$([ "$name" = "train" ] && echo '3.3 GB, 800' || echo '0.4 GB, 100') images) ..."
    wget -q --show-progress "$url" -O "$zip_path"

    info "Extracting $name ..."
    unzip -q -o "$zip_path" -d "$ROOT_DIR/datasets/"

    rm -f "$zip_path"
    ok "DIV2K $name ready ($(count_images "$dest_dir") images in $dest_dir/)."
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    info "Nexus-Steg dataset download — DIV2K (train + valid)"
    echo ""

    download_div2k_split "train" "$DIV2K_TRAIN_URL" "$TRAIN_DIR"
    echo ""
    download_div2k_split "valid" "$DIV2K_VALID_URL" "$VALID_DIR"

    echo ""
    ok "All datasets ready."
    echo ""
    info "Final counts:"
    info "  Train: $(count_images "$TRAIN_DIR") images"
    info "  Valid: $(count_images "$VALID_DIR") images"
}

main "$@"
