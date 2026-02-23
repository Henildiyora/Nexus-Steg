#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COVER_DIR="$ROOT_DIR/datasets/cover"
SECRET_TRAIN_DIR="$ROOT_DIR/datasets/secret/train"
SECRET_TEST_DIR="$ROOT_DIR/datasets/secret/test"

COCO_URL="http://images.cocodataset.org/zips/val2017.zip"

S3_BASE="https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/tarballs"

SN2_TRAIN_TARBALLS=(
    "SN2_buildings_train_AOI_2_Vegas.tar.gz"
)

SN2_TEST_TARBALLS=(
    "AOI_2_Vegas_Test_public.tar.gz"
)

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
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
           -o -iname '*.tif' -o -iname '*.tiff' \) 2>/dev/null | wc -l
}

# ── MS-COCO val2017 (cover images) ──────────────────────────────────────────

download_coco() {
    info "Checking cover images in $COVER_DIR ..."
    mkdir -p "$COVER_DIR"

    existing=$(count_images "$COVER_DIR")
    if [ "$existing" -ge 100 ]; then
        ok "Cover directory already has $existing images — skipping download."
        return
    fi

    need_cmd curl
    need_cmd unzip

    local zip_path="$ROOT_DIR/datasets/val2017.zip"

    info "Downloading MS-COCO val2017 (~1 GB) ..."
    curl -L --progress-bar -o "$zip_path" "$COCO_URL"

    info "Extracting to $COVER_DIR ..."
    unzip -q -o "$zip_path" -d "$ROOT_DIR/datasets/"

    if [ -d "$ROOT_DIR/datasets/val2017" ]; then
        mv "$ROOT_DIR/datasets/val2017"/* "$COVER_DIR/" 2>/dev/null || true
        rmdir "$ROOT_DIR/datasets/val2017" 2>/dev/null || true
    fi

    rm -f "$zip_path"
    ok "Cover images ready ($(count_images "$COVER_DIR") files)."
}

# ── SpaceNet 2 — all 4 cities (secret images) ───────────────────────────────

download_sn2() {
    local split_name="$1" dest_dir="$2"
    shift 2
    local tarballs=("$@")

    info "Checking secret $split_name images in $dest_dir ..."
    mkdir -p "$dest_dir"

    existing=$(count_images "$dest_dir")
    if [ "$existing" -ge 100 ]; then
        ok "Secret $split_name directory already has $existing images — skipping download."
        return
    fi

    need_cmd curl

    for tarball in "${tarballs[@]}"; do
        local url="$S3_BASE/$tarball"
        local archive_path="$ROOT_DIR/datasets/$tarball"

        info "Downloading $tarball ..."
        curl -L --progress-bar -o "$archive_path" "$url"

        info "Extracting $tarball ..."
        tar -xzf "$archive_path" -C "$dest_dir"

        rm -f "$archive_path"
        ok "$tarball done."
    done

    ok "Secret $split_name images ready ($(count_images "$dest_dir") files)."
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    info "Nexus-Steg dataset download"
    info "SpaceNet 2 — Vegas"
    echo ""

    download_coco
    echo ""
    download_sn2 "train" "$SECRET_TRAIN_DIR" "${SN2_TRAIN_TARBALLS[@]}"
    echo ""
    download_sn2 "test"  "$SECRET_TEST_DIR"  "${SN2_TEST_TARBALLS[@]}"

    echo ""
    ok "All datasets ready."
}

main "$@"
