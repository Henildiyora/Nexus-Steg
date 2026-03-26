#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COVER_DIR="$ROOT_DIR/datasets/cover"
SECRET_TRAIN_DIR="$ROOT_DIR/datasets/secret/train"
SECRET_TEST_DIR="$ROOT_DIR/datasets/secret/test"

COCO_URL="http://images.cocodataset.org/zips/val2017.zip"

S3_BUCKET="s3://spacenet-dataset/spacenet/SN2_buildings"

CITIES=("AOI_2_Vegas" "AOI_3_Paris" "AOI_4_Shanghai" "AOI_5_Khartoum")

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

# ── SpaceNet 2 — MUL-PanSharpen only, all 4 cities ─────────────────────────
#
# S3 layout (no account needed with --no-sign-request):
#   s3://spacenet-dataset/spacenet/SN2_buildings/train/<CITY>/PS-MS/
#   s3://spacenet-dataset/spacenet/SN2_buildings/test_public/<CITY>/PS-MS/
#
# Local layout:
#   datasets/secret/train/<CITY>/   ← MUL-PanSharpen TIFs
#   datasets/secret/test/<CITY>/

download_sn2_city() {
    local split_name="$1" s3_split="$2" dest_dir="$3" city="$4"
    local city_dir="$dest_dir/$city"

    mkdir -p "$city_dir"

    existing=$(count_images "$city_dir")
    if [ "$existing" -ge 10 ]; then
        ok "$city ($split_name) already has $existing images — skipping."
        return
    fi

    local s3_path="$S3_BUCKET/$s3_split/$city/PS-MS/"

    info "Downloading $city $split_name MUL-PanSharpen ..."
    aws s3 cp "$s3_path" "$city_dir/" \
        --recursive --no-sign-request --quiet

    ok "$city ($split_name) done — $(count_images "$city_dir") images."
}

download_sn2() {
    need_cmd aws

    info "Downloading SpaceNet 2 MUL-PanSharpen (4 cities, train + test) ..."
    echo ""

    for city in "${CITIES[@]}"; do
        download_sn2_city "train" "train" "$SECRET_TRAIN_DIR" "$city"
    done

    echo ""

    for city in "${CITIES[@]}"; do
        download_sn2_city "test" "test_public" "$SECRET_TEST_DIR" "$city"
    done
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    info "Nexus-Steg dataset download"
    info "SpaceNet 2 — Vegas, Paris, Shanghai, Khartoum (MUL-PanSharpen)"
    echo ""

    download_coco
    echo ""
    download_sn2

    echo ""
    ok "All datasets ready."
    echo ""
    info "Final counts:"
    info "  Cover:        $(count_images "$COVER_DIR") images"
    info "  Secret train: $(count_images "$SECRET_TRAIN_DIR") images"
    info "  Secret test:  $(count_images "$SECRET_TEST_DIR") images"
}

main "$@"
