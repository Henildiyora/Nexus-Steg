# ── Nexus-Steg Makefile ──────────────────────────────────────────────────────

PYTHON     ?= uv run python
EPOCHS     ?= 100
BATCH_SIZE ?=
CHECKPOINT ?= checkpoints/nexus_epoch_99.pth
COVER_DIR  ?= datasets/cover
SECRET_DIR ?= datasets/secret/test

# ── Phony targets ────────────────────────────────────────────────────────────

.PHONY: all install download train test visualize health clean help

all: install download train test

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv sync

download:
	bash scripts/download_data.sh

# ── Training ─────────────────────────────────────────────────────────────────

train:
	$(PYTHON) main.py

# ── Evaluation ───────────────────────────────────────────────────────────────

test:
	$(PYTHON) evaluate.py \
		--checkpoint $(CHECKPOINT) \
		--cover_dir $(COVER_DIR) \
		--secret_dir $(SECRET_DIR)

# ── Visualization ────────────────────────────────────────────────────────────

visualize:
	$(PYTHON) visualize_arch.py

# ── Health check ─────────────────────────────────────────────────────────────

health:
	$(PYTHON) check_health.py

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	rm -rf results/ checkpoints/

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo "Nexus-Steg Makefile targets:"
	@echo ""
	@echo "  make install     Install Python dependencies (uv sync)"
	@echo "  make download    Download MS-COCO and SpaceNet datasets"
	@echo "  make train       Train the model (EPOCHS=$(EPOCHS))"
	@echo "  make test        Evaluate against attack suite (CHECKPOINT=$(CHECKPOINT))"
	@echo "  make visualize   Generate architecture graph PNGs"
	@echo "  make health      Run architecture health check"
	@echo "  make clean       Remove results/ and checkpoints/"
	@echo "  make all         install -> download -> train -> test"
	@echo ""
	@echo "Override variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  EPOCHS=$(EPOCHS)  CHECKPOINT=$(CHECKPOINT)"
	@echo "  COVER_DIR=$(COVER_DIR)  SECRET_DIR=$(SECRET_DIR)"
