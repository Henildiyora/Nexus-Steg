# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Nexus-Steg Makefile                                                       ║
# ║  Full pipeline: install → download → verify → sanity → train → evaluate    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Configuration (override on CLI: make train EPOCHS=50 BATCH_SIZE=32) ──────
PYTHON           ?= python
PIP              ?= pip
EPOCHS           ?= 100
BATCH_SIZE       ?= 64
PATIENCE         ?= 15
CHECKPOINT_EVERY ?= 10
SEED             ?= 42
NUM_WORKERS      ?=

# SpaceNet 2 cities to download (space-separated, subset to save disk)
#   All four: Vegas Paris Shanghai Khartoum  (~56 GB download, ~10k images)
#   Minimal:  Vegas Paris                    (~28 GB download, ~5k images)
AOIS ?= Vegas Paris Shanghai Khartoum

# ── Directories ──────────────────────────────────────────────────────────────
COVER_DIR  := datasets/cover
SECRET_DIR := datasets/secret/MUL-PanSharpen
RESULTS    := results
CKPT_DIR   := checkpoints
STAMPS     := .stamps
TMP_DIR    := /tmp/nexus-steg-dl

# ── URLs ─────────────────────────────────────────────────────────────────────
COCO_VAL_URL  := http://images.cocodataset.org/zips/val2017.zip
COCO_TEST_URL := http://images.cocodataset.org/zips/test2017.zip

SN2_BUCKET := s3://spacenet-dataset/spacenet/SN2_buildings

# AOI city → S3 path component
SN2_AOI_Vegas    := AOI_2_Vegas
SN2_AOI_Paris    := AOI_3_Paris
SN2_AOI_Shanghai := AOI_4_Shanghai
SN2_AOI_Khartoum := AOI_5_Khartoum

# ── Derived paths ────────────────────────────────────────────────────────────
BEST_CKPT   := $(CKPT_DIR)/nexus_best.pth
TRAIN_LOG   := $(RESULTS)/training_log.csv
EVAL_REPORT := $(RESULTS)/evaluation/report.txt

# Stamp files (track completed steps, avoid redundant work)
STAMP_INSTALL   := $(STAMPS)/.installed
STAMP_COCO_VAL  := $(STAMPS)/.coco_val2017
STAMP_COCO_TEST := $(STAMPS)/.coco_test2017
SN2_STAMPS      := $(foreach city,$(AOIS),$(STAMPS)/.sn2_$(city))

# Build NUM_WORKERS flag only if set
ifdef NUM_WORKERS
  WORKER_FLAG := --num_workers $(NUM_WORKERS)
else
  WORKER_FLAG :=
endif

# ── Phony targets ────────────────────────────────────────────────────────────
.PHONY: all install data covers secrets verify health sanity overfit train \
        evaluate plot clean clean-data clean-stamps clean-all help

# ══════════════════════════════════════════════════════════════════════════════
# High-level targets
# ══════════════════════════════════════════════════════════════════════════════

all: evaluate
	@echo ""
	@echo "Pipeline complete. Results in $(RESULTS)/, checkpoints in $(CKPT_DIR)/."

help:
	@echo "Nexus-Steg Makefile"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  make install       Install project (pip install -e .)"
	@echo "  make data          Download all datasets (covers + secrets)"
	@echo "  make covers        Download COCO val2017 + test2017 covers"
	@echo "  make secrets       Download SpaceNet 2 satellite secrets"
	@echo "  make verify        Count images in dataset directories"
	@echo "  make health        Run architecture health check"
	@echo "  make sanity        Run sanity check (verify initial losses)"
	@echo "  make overfit       Overfit on one batch (verify capacity)"
	@echo "  make train         Full training ($(EPOCHS) epochs)"
	@echo "  make evaluate      Evaluate best checkpoint against attacks"
	@echo "  make plot          Plot training curves from CSV log"
	@echo "  make all           Run the entire pipeline end-to-end"
	@echo ""
	@echo "  make clean         Remove results and checkpoints"
	@echo "  make clean-stamps  Remove download stamps (re-trigger downloads)"
	@echo "  make clean-data    Remove downloaded datasets"
	@echo "  make clean-all     Remove everything (data + results + stamps)"
	@echo ""
	@echo "Configuration (override via CLI):"
	@echo "  EPOCHS=$(EPOCHS)  BATCH_SIZE=$(BATCH_SIZE)  PATIENCE=$(PATIENCE)"
	@echo "  AOIS=\"$(AOIS)\""
	@echo ""
	@echo "Examples:"
	@echo "  make train EPOCHS=50 BATCH_SIZE=32"
	@echo "  make data AOIS=\"Vegas Paris\""
	@echo "  make all NUM_WORKERS=4"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Install dependencies
# ══════════════════════════════════════════════════════════════════════════════

install: $(STAMP_INSTALL)

$(STAMP_INSTALL): pyproject.toml | $(STAMPS)
	$(PIP) install -e . -q
	@touch $@
	@echo "[OK] Dependencies installed."

# ══════════════════════════════════════════════════════════════════════════════
# 2. Download datasets
# ══════════════════════════════════════════════════════════════════════════════

data: covers secrets
	@echo "[OK] All datasets downloaded."

# ── COCO cover images (val2017 + test2017 → ~46,000 images) ─────────────────

covers: $(STAMP_COCO_VAL) $(STAMP_COCO_TEST)
	@echo "[OK] Cover images ready in $(COVER_DIR)/"

$(STAMP_COCO_VAL): | $(STAMPS)
	@mkdir -p $(COVER_DIR) $(TMP_DIR)
	@echo "[INFO] Downloading COCO val2017 (~1 GB, 5,000 images)..."
	wget -q --show-progress $(COCO_VAL_URL) -O $(TMP_DIR)/val2017.zip
	@echo "[INFO] Extracting val2017..."
	unzip -q -o $(TMP_DIR)/val2017.zip -d $(TMP_DIR)/coco_tmp
	mv $(TMP_DIR)/coco_tmp/val2017/* $(COVER_DIR)/
	rm -rf $(TMP_DIR)/coco_tmp $(TMP_DIR)/val2017.zip
	@touch $@
	@echo "[OK] COCO val2017 done."

$(STAMP_COCO_TEST): | $(STAMPS)
	@mkdir -p $(COVER_DIR) $(TMP_DIR)
	@echo "[INFO] Downloading COCO test2017 (~6.2 GB, 40,670 images)..."
	wget -q --show-progress $(COCO_TEST_URL) -O $(TMP_DIR)/test2017.zip
	@echo "[INFO] Extracting test2017..."
	unzip -q -o $(TMP_DIR)/test2017.zip -d $(TMP_DIR)/coco_tmp
	mv $(TMP_DIR)/coco_tmp/test2017/* $(COVER_DIR)/
	rm -rf $(TMP_DIR)/coco_tmp $(TMP_DIR)/test2017.zip
	@touch $@
	@echo "[OK] COCO test2017 done."

# ── SpaceNet 2 secret images (satellite TIFFs via AWS CLI) ───────────────────

secrets: $(SN2_STAMPS)
	@echo "[OK] Secret images ready in $(SECRET_DIR)/"

# Generic rule: aws s3 sync MUL-PanSharpen TIFFs for each city (no account needed)
define SN2_RULE
$(STAMPS)/.sn2_$(1): | $(STAMPS)
	@mkdir -p $(SECRET_DIR)
	@echo "[INFO] Syncing SpaceNet 2 — $(1) MUL-PanSharpen via AWS CLI ..."
	aws s3 sync \
		"$(SN2_BUCKET)/train/$(SN2_AOI_$(1))/MUL-PanSharpen/" \
		"$(SECRET_DIR)/" \
		--no-sign-request --quiet
	@touch $$@
	@echo "[OK] SpaceNet 2 $(1) done."
endef

$(foreach city,$(AOIS),$(eval $(call SN2_RULE,$(city))))

# ══════════════════════════════════════════════════════════════════════════════
# 3. Verify datasets
# ══════════════════════════════════════════════════════════════════════════════

verify: data
	@echo "════════════════════════════════════════════════════════"
	@echo "  Dataset Verification"
	@echo "════════════════════════════════════════════════════════"
	@echo "  Cover images:  $$(find $(COVER_DIR) -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' \) 2>/dev/null | wc -l)"
	@echo "  Secret images: $$(find $(SECRET_DIR) -type f -name '*.tif' 2>/dev/null | wc -l)"
	@echo "════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Health check
# ══════════════════════════════════════════════════════════════════════════════

health: $(STAMP_INSTALL)
	$(PYTHON) check_health.py

# ══════════════════════════════════════════════════════════════════════════════
# 5. Sanity check (verify initial losses)
# ══════════════════════════════════════════════════════════════════════════════

sanity: $(STAMP_INSTALL) data
	$(PYTHON) main.py --sanity --seed $(SEED) $(WORKER_FLAG)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Overfit one batch
# ══════════════════════════════════════════════════════════════════════════════

overfit: $(STAMP_INSTALL) data
	$(PYTHON) main.py --overfit_one_batch --seed $(SEED) $(WORKER_FLAG)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Full training
# ══════════════════════════════════════════════════════════════════════════════

train: $(BEST_CKPT)

$(BEST_CKPT): $(STAMP_INSTALL) data
	@mkdir -p $(CKPT_DIR) $(RESULTS)
	$(PYTHON) main.py \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--checkpoint_every $(CHECKPOINT_EVERY) \
		--patience $(PATIENCE) \
		--seed $(SEED) \
		$(WORKER_FLAG)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Evaluate
# ══════════════════════════════════════════════════════════════════════════════

evaluate: $(EVAL_REPORT)

$(EVAL_REPORT): $(BEST_CKPT)
	@mkdir -p $(RESULTS)/evaluation
	$(PYTHON) evaluate.py --checkpoint $(BEST_CKPT)
	@echo ""
	@echo "════════════════════════════════════════════════════════"
	@cat $(EVAL_REPORT)
	@echo "════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
# 9. Plot training curves (requires training_log.csv)
# ══════════════════════════════════════════════════════════════════════════════

plot: $(TRAIN_LOG)
	@$(PYTHON) -c "\
import pandas as pd; \
import matplotlib; matplotlib.use('Agg'); \
import matplotlib.pyplot as plt; \
df = pd.read_csv('$(TRAIN_LOG)'); \
fig, axes = plt.subplots(2, 2, figsize=(16, 10)); \
fig.suptitle('Nexus-Steg Training Curves', fontsize=16); \
ax = axes[0,0]; ax.plot(df['epoch'], df['val_psnr_stego'], label='PSNR(stego)'); ax.plot(df['epoch'], df['val_psnr_secret'], label='PSNR(secret)'); ax.axvline(30, color='gray', ls='--', alpha=.5); ax.axvline(60, color='gray', ls=':', alpha=.5); ax.set(xlabel='Epoch', ylabel='PSNR (dB)', title='Validation PSNR'); ax.legend(); ax.grid(True, alpha=.3); \
ax = axes[0,1]; ax.plot(df['epoch'], df['val_ssim_stego'], label='SSIM(stego)'); ax.plot(df['epoch'], df['val_ssim_secret'], label='SSIM(secret)'); ax.axvline(30, color='gray', ls='--', alpha=.5); ax.axvline(60, color='gray', ls=':', alpha=.5); ax.set(xlabel='Epoch', ylabel='SSIM', title='Validation SSIM'); ax.legend(); ax.grid(True, alpha=.3); \
ax = axes[1,0]; ax.plot(df['epoch'], df['l_inv'], label='l_inv'); ax.plot(df['epoch'], df['l_rec'], label='l_rec'); ax.plot(df['epoch'], df['l_disc'], label='l_disc'); ax.axvline(30, color='gray', ls='--', alpha=.5); ax.axvline(60, color='gray', ls=':', alpha=.5); ax.set(xlabel='Epoch', ylabel='Loss', title='Training Losses'); ax.legend(); ax.grid(True, alpha=.3); \
ax = axes[1,1]; ax.plot(df['epoch'], df['lr'], color='tab:orange'); ax.axvline(30, color='gray', ls='--', alpha=.5); ax.axvline(60, color='gray', ls=':', alpha=.5); ax.set(xlabel='Epoch', ylabel='LR', title='Cosine Annealing LR'); ax.grid(True, alpha=.3); \
plt.tight_layout(); plt.savefig('$(RESULTS)/training_curves.png', dpi=150); \
print('Saved to $(RESULTS)/training_curves.png')"

# ══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════════

clean:
	rm -rf $(RESULTS) $(CKPT_DIR)
	@echo "[OK] Removed results and checkpoints."

clean-stamps:
	rm -rf $(STAMPS)
	@echo "[OK] Removed stamps (downloads will re-trigger)."

clean-data:
	rm -rf datasets
	@echo "[OK] Removed datasets."

clean-all: clean clean-stamps clean-data
	rm -rf $(TMP_DIR)
	@echo "[OK] Full cleanup complete."

# ── Create stamp directory ───────────────────────────────────────────────────

$(STAMPS):
	@mkdir -p $(STAMPS)
