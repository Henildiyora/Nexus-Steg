# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Nexus-Steg Makefile                                                       ║
# ║  Full pipeline: install → download → verify → sanity → train → evaluate    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Configuration (override on CLI: make train EPOCHS=50 BATCH_SIZE=32) ──────
PYTHON           ?= python
PIP              ?= pip
EPOCHS           ?= 100
BATCH_SIZE       ?= 16
PATIENCE         ?= 15
CHECKPOINT_EVERY ?= 10
SEED             ?= 42
NUM_WORKERS      ?=

# ── Directories ──────────────────────────────────────────────────────────────
TRAIN_DIR  := datasets/DIV2K_train_HR
VAL_DIR    := datasets/DIV2K_valid_HR
RESULTS    := results
CKPT_DIR   := checkpoints
STAMPS     := .stamps
TMP_DIR    := /tmp/nexus-steg-dl

# ── URLs ─────────────────────────────────────────────────────────────────────
DIV2K_TRAIN_URL := http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
DIV2K_VALID_URL := http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

# ── Derived paths ────────────────────────────────────────────────────────────
BEST_CKPT   := $(CKPT_DIR)/nexus_best.pth
TRAIN_LOG   := $(RESULTS)/training_log.csv
EVAL_REPORT := $(RESULTS)/evaluation/report.txt

# Stamp files (track completed steps, avoid redundant work)
STAMP_INSTALL    := $(STAMPS)/.installed
STAMP_DIV2K_TRAIN := $(STAMPS)/.div2k_train
STAMP_DIV2K_VALID := $(STAMPS)/.div2k_valid

# Build NUM_WORKERS flag only if set
ifdef NUM_WORKERS
  WORKER_FLAG := --num_workers $(NUM_WORKERS)
else
  WORKER_FLAG :=
endif

# ── Phony targets ────────────────────────────────────────────────────────────
.PHONY: all install data verify health sanity overfit train \
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
	@echo "  make data          Download DIV2K dataset (train + valid)"
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
	@echo ""
	@echo "Examples:"
	@echo "  make train EPOCHS=50 BATCH_SIZE=32"
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
# 2. Download DIV2K dataset
# ══════════════════════════════════════════════════════════════════════════════

data: $(STAMP_DIV2K_TRAIN) $(STAMP_DIV2K_VALID)
	@echo "[OK] DIV2K dataset ready."

$(STAMP_DIV2K_TRAIN): | $(STAMPS)
	@mkdir -p datasets $(TMP_DIR)
	@echo "[INFO] Downloading DIV2K train HR (~3.3 GB, 800 images)..."
	wget -q --show-progress $(DIV2K_TRAIN_URL) -O $(TMP_DIR)/DIV2K_train_HR.zip
	@echo "[INFO] Extracting DIV2K train..."
	unzip -q -o $(TMP_DIR)/DIV2K_train_HR.zip -d datasets/
	rm -f $(TMP_DIR)/DIV2K_train_HR.zip
	@touch $@
	@echo "[OK] DIV2K train done (800 images in $(TRAIN_DIR)/)."

$(STAMP_DIV2K_VALID): | $(STAMPS)
	@mkdir -p datasets $(TMP_DIR)
	@echo "[INFO] Downloading DIV2K valid HR (~0.4 GB, 100 images)..."
	wget -q --show-progress $(DIV2K_VALID_URL) -O $(TMP_DIR)/DIV2K_valid_HR.zip
	@echo "[INFO] Extracting DIV2K valid..."
	unzip -q -o $(TMP_DIR)/DIV2K_valid_HR.zip -d datasets/
	rm -f $(TMP_DIR)/DIV2K_valid_HR.zip
	@touch $@
	@echo "[OK] DIV2K valid done (100 images in $(VAL_DIR)/)."

# ══════════════════════════════════════════════════════════════════════════════
# 3. Verify datasets
# ══════════════════════════════════════════════════════════════════════════════

verify: data
	@echo "════════════════════════════════════════════════════════"
	@echo "  Dataset Verification"
	@echo "════════════════════════════════════════════════════════"
	@echo "  Train images: $$(find $(TRAIN_DIR) -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l)"
	@echo "  Valid images: $$(find $(VAL_DIR) -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l)"
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
