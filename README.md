# Nexus-Steg

**A Robust Semantic-Texture Hybrid Steganography System**

Nexus-Steg hides secret images inside cover images using a U-Net + Vision Transformer encoder with CBAM attention, adversarial steganalysis training, and differentiable noise simulation. The stego image survives JPEG compression, blur, resize, and AI-based detection.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Henildiyora/Nexus-Steg.git
cd Nexus-Steg
uv sync

# 2. Place your images
#    datasets/cover/          <- cover images (PNG/JPG)
#    datasets/secret/MUL-PanSharpen/  <- secret images (TIFF/PNG/JPG)

# 3. Run sanity check first (always)
python main.py --sanity

# 4. Test model capacity
python main.py --overfit_one_batch

# 5. Train
python main.py --epochs 100

# 6. Evaluate
python evaluate.py --checkpoint checkpoints/nexus_best.pth
```

---

## What's New in This Branch

This update applies Karpathy's "A Recipe for Training Neural Networks" to make training reproducible, debuggable, and robust:

| Feature | What It Does |
|---------|-------------|
| **Reproducibility seeds** | `torch.manual_seed(42)` + numpy + random + cudnn deterministic. Same results every run. |
| **`--sanity` mode** | Verifies initial losses match expected values and saves input visualization before training |
| **`--overfit_one_batch`** | Trains on 1 batch for 200 steps to verify the model has enough capacity |
| **Weight decay** | `1e-5` on generator optimizer to prevent weight explosion |
| **Early stopping** | Stops training if PSNR(secret) doesn't improve for 15 epochs. Saves best checkpoint. |
| **CSV logging** | Writes all metrics to `results/training_log.csv` every epoch for easy plotting |
| **CLI arguments** | All settings configurable via command line (`--epochs`, `--batch_size`, `--patience`, `--seed`) |

---

## Step-by-Step Training Guide

### Step 1: Sanity Check

```bash
python main.py --sanity
```

**What to check:**

| Output | Expected | Problem If Wrong |
|--------|----------|-----------------|
| Cover range | `[-1.000, 1.000]` | Data normalization broken |
| Secret range | `[-1.000, 1.000]` | TIFF loading broken |
| `l_inv` | 0.01 - 0.10 | Encoder init is off |
| `l_rec` | 0.30 - 0.70 | Reveal network is off |
| `l_disc` | ~0.693 | Discriminator init is off |

Also open `results/sanity_inputs.png` -- you should see real cover images (top row) and secret images (bottom row). If they look garbled, the data pipeline has a bug.

### Step 2: Overfit One Batch

```bash
python main.py --overfit_one_batch
```

**What to check:**

- **PASS** (loss < 0.01): Model has sufficient capacity. Proceed to training.
- **WARN** (loss 0.01 - 0.10): Likely fine but monitor recovery during training.
- **FAIL** (loss > 0.10): Architecture or learning rate problem. Do not proceed.

### Step 3: Train

```bash
# Colab / A100 / H100:
python main.py --epochs 100 --batch_size 64

# Mac (MPS):
python main.py --epochs 100 --batch_size 4

# All options:
python main.py --epochs 100 --batch_size 64 --patience 15 --checkpoint_every 10 --seed 42
```

Training runs in 3 phases automatically:

| Phase | Epochs | What Happens | What to Watch |
|-------|--------|-------------|---------------|
| **1** | 0-29 | Pure hiding/recovery, no noise, no adversarial | `rec` should drop steadily. PSNR(secret) should climb to 25-30dB. |
| **2** | 30-59 | Noise layer ON, adversarial ON (mild) | PSNR(secret) will drop at epoch 30 (normal!), then recover over ~10 epochs. |
| **3** | 60-99 | Full adversarial pressure, max recovery weight | Metrics should plateau. `disc` near 0.693 means stego is undetectable. |

**Target metrics at end of training:**

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| PSNR(stego) | > 30dB | > 33dB | > 36dB |
| SSIM(stego) | > 0.90 | > 0.95 | > 0.97 |
| PSNR(secret) | > 24dB | > 28dB | > 32dB |
| SSIM(secret) | > 0.55 | > 0.70 | > 0.85 |
| disc | 0.60-0.75 | 0.65-0.72 | ~0.693 |

**Red flags during training:**
- `rec` stays at 0.000 --> network is not learning to recover the secret
- PSNR(secret) > 50dB in early epochs --> network is outputting the secret directly (not hiding anything)
- `disc` drops below 0.3 and stays --> discriminator is winning, stego images are too obvious
- Metrics flat for 15 epochs --> early stopping will trigger automatically

### Step 4: Check Outputs

**Visual results:** Open `results/epoch_N.png` -- each image shows `[cover | secret | stego | revealed]` side by side. Cover and stego should look identical. Secret and revealed should look similar by epoch 50+.

**Training log:** Open `results/training_log.csv` to plot training curves. Columns: epoch, phase, train_loss, l_inv, l_rec, l_disc, val_psnr_stego, val_ssim_stego, val_psnr_secret, val_ssim_secret, lr.

**Best checkpoint:** `checkpoints/nexus_best.pth` is automatically saved whenever PSNR(secret) improves.

### Step 5: Evaluate

```bash
python evaluate.py --checkpoint checkpoints/nexus_best.pth
```

Runs 8 attack tests (JPEG-90, JPEG-50, blur, noise, resize, social media simulation, steganalysis detection) and produces a PASS/WARN/FAIL report with visual evidence saved to `results/evaluation/`.

```bash
# Test on completely unseen images:
python evaluate.py --checkpoint checkpoints/nexus_best.pth \
    --cover_dir path/to/new/covers --secret_dir path/to/new/secrets
```

---

## Project Structure

```
nexus-steg/
├── main.py                  # Training entry point (sanity, overfit, train modes)
├── evaluate.py              # Post-training evaluation test suite (8 attack scenarios)
├── visualize_arch.py        # Generate torchviz computational graphs
├── src/
│   ├── core/device.py       # Hardware detection (CUDA / MPS / CPU)
│   ├── data/pipeline.py     # Dataset loading, TIFF support, train/val split
│   ├── engine/trainer.py    # Training loop, losses, validation, metrics
│   └── models/
│       ├── hybrid_transformer.py  # HidingNetwork + RevealNetwork (U-Net + ViT + CBAM)
│       ├── noise_layer.py         # Differentiable JPEG, blur, noise, dropout, resize
│       └── discriminator.py       # SRNet-inspired steganalysis discriminator
├── datasets/
│   ├── cover/               # Cover images (PNG/JPG)
│   └── secret/MUL-PanSharpen/  # Secret images (TIFF/PNG/JPG)
├── checkpoints/             # Saved model weights
└── results/                 # Visual outputs, training_log.csv, evaluation reports
```

---

## CLI Reference

```bash
# Full training
python main.py --epochs 100 --batch_size 64 --checkpoint_every 10 --patience 15 --seed 42

# Sanity check only
python main.py --sanity

# Overfit one batch only
python main.py --overfit_one_batch

# Evaluate trained model
python evaluate.py --checkpoint checkpoints/nexus_best.pth
python evaluate.py --checkpoint checkpoints/nexus_epoch_99.pth --cover_dir path/to/covers --secret_dir path/to/secrets
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | auto (64 CUDA, 4 MPS) | Batch size |
| `--checkpoint_every` | 10 | Save checkpoint every N epochs |
| `--patience` | 15 | Early stopping patience |
| `--seed` | 42 | Random seed for reproducibility |
| `--sanity` | off | Run sanity checks only |
| `--overfit_one_batch` | off | Run capacity test only |

---

## Architecture Overview

```
Cover (3ch) + Secret (3ch)
        |
    [CONCAT -> 6ch]
        |
    U-Net Encoder (ResBlock+CBAM at each level)
    64 -> 128 -> 256 channels, pooling 2x at each step
        |
    ViT Bottleneck (256ch, 32x32 = 1024 tokens, 8-head attention)
        |
    U-Net Decoder (skip connections from encoder)
    256 -> 128 -> 64 channels, upsample 2x at each step
        |
    Conv 1x1 + tanh -> residual
        |
    Stego = Cover + alpha * residual     (alpha is learnable, starts at 0.4)
        |
    [Noise Layer - training only]         (JPEG, blur, resize, noise, dropout)
        |
    Reveal Network (same U-Net + ViT architecture, 3ch input)
        |
    Recovered Secret (3ch)
```

---

## Dataset Recommendations

| Size | Training Pairs | Effect |
|------|---------------|--------|
| 1,000 | 800 | Model memorizes textures. Poor generalization. |
| 5,000 | 4,000 | Minimum viable. Learns basic patterns. |
| **10,000** | **8,000** | **Sweet spot. Strong generalization for this architecture.** |
| 50,000+ | 40,000+ | Diminishing returns. Only for production. |

Cover images: [MS-COCO](https://cocodataset.org/) train2017 or val2017.
Secret images: [SpaceNet](https://spacenet.ai/) satellite TIFF, or any other images.

---

## Requirements

- Python 3.11+
- PyTorch 2.10+
- CUDA (recommended), MPS (Mac), or CPU
- See `pyproject.toml` for full dependency list
