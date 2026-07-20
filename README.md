# Conditional DCGAN — Anime Face Generation

Conditional DCGAN that generates anime faces conditioned on **hair color** and **eye color** labels.

## Requirements

```bash
pip install torch torchvision pandas pillow tqdm
```
GPU (CUDA) recommended but not required — falls back to CPU automatically (mixed precision only activates on CUDA).

## Project Structure

```
project/
├── source/
│   ├── config.py     # hyperparameters, paths, device settings
│   ├── dataset.py     # PyTorch Dataset — loads images + labels.csv
│   ├── models.py       # Generator / Discriminator architectures
│   ├── train.py        # training loop
│   └── generate.py     # inference / image generation
├── dataset/
│   ├── images/          # source images
│   └── labels.csv        # columns: image, hair, eyes
├── models/                # checkpoints (auto-created)
└── outputs/                # loss log, preview grids, generated images (auto-created)
```

Before training: put images in `dataset/images/` and a `dataset/labels.csv` with columns `image, hair, eyes` (e.g. `img_001.png, blonde, blue`).

---

## `config.py`

Central place for every setting; all other files import from it.

| Setting | Purpose |
|---|---|
| `IMAGE_SIZE`, `IMAGE_CHANNELS` | images resized to 64×64 RGB |
| `NUM_HAIR_CLASSES`, `NUM_EYE_CLASSES` | must be ≥ the number of distinct labels actually in `labels.csv` |
| `LATENT_DIM` | size of the random noise vector fed to the Generator |
| `FEATURE_MAPS` | base channel width for both networks |
| `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `BETA1`/`BETA2` | Adam optimizer training settings |
| `EMA_DECAY` | target decay for the EMA generator (ramps up to this over training, doesn't start here) |
| `DEVICE` | auto-picks CUDA if available |
| `USE_AMP` | mixed-precision training toggle (auto: on for CUDA, off for CPU) |
| `CUDNN_BENCHMARK` | speeds up training since input size is fixed |
| `SAVE_EVERY` | checkpoint + preview-image frequency (epochs) |
| `GENERATOR_PATH` / `DISCRIMINATOR_PATH` / `EMA_GENERATOR_PATH` | checkpoint file locations |

No usage of its own — just imported (`from config import *`) by every other file.

---

## `dataset.py`

Defines `AnimeDataset`, a PyTorch `Dataset`.

**What it does:**
- Reads `labels.csv`, builds sorted hair/eye label → integer index mappings.
- Loads each image, resizes to `IMAGE_SIZE`, applies `RandomHorizontalFlip`, converts to tensor, normalizes to `[-1, 1]` (matches the Generator's `Tanh` output range).
- Returns `(image_tensor, hair_label, eye_label)` per item.
- Skips unreadable/corrupted images automatically (bounded retry — won't infinite-loop even if every image happened to fail).

**Usage / features:**
```bash
python source/dataset.py
```
Standalone run prints a summary: total images, label counts, the hair/eye label→index mapping, and one sample's shape — useful as a quick sanity check that your CSV and images directory are set up correctly before training.

---

## `models.py`

Defines the two networks.

**`Generator`** — takes a noise vector + hair/eye labels, outputs a 64×64 RGB image.
- Labels go through embedding layers, concatenated with the noise vector.
- 4 upsampling blocks (`ConvTranspose2d` + `BatchNorm` + `ReLU`) taking it from 1×1 → 64×64.
- A self-attention layer at the 16×16 stage — lets distant pixels (e.g. both eyes, hair strands) influence each other, not just nearby pixels.
- Final `Tanh` activation, output range `[-1, 1]`.

**`Discriminator`** — takes an image + hair/eye labels, outputs a raw score (logit) for real-vs-fake.
- Labels are embedded, reshaped into full-size 64×64 "condition maps," `tanh`-bounded, and concatenated onto the image as extra channels.
- 4 downsampling blocks using **spectral normalization** (rather than BatchNorm) for training stability.
- A matching self-attention layer at the 16×16 stage.
- Outputs a raw logit — pair with `BCEWithLogitsLoss`, not `BCELoss`.

**`weights_init`** — applies DCGAN-standard weight initialization to conv/linear/embedding/batchnorm layers (correctly handles spectral-norm-wrapped layers too).

**Usage / features:**
```bash
python source/models.py
```
Standalone run builds both networks, runs one dummy batch through each, and asserts the output shapes/values are sane — a fast way to check the architecture is wired correctly before a full training run.

---

## `train.py`

The training loop.

**Key features:**
- Loads existing checkpoints automatically if present (resumes training).
- **Mixed precision (AMP)** — faster training on CUDA.
- **DiffAugment** — differentiable random translation + brightness/contrast jitter applied identically to real and fake images before the Discriminator sees them; reduces D overfitting on small/medium datasets.
- **EMA generator** — a second copy of the Generator holding a running average of its weights, generally cleaner/more stable than the raw weights. Decay ramps up gradually rather than starting at full strength, so it doesn't lag behind on shorter training runs.
- **Label smoothing** — real-image target is 0.9, not 1.0, to keep the Discriminator from getting overconfident.
- Real + fake batches are combined into a single Discriminator forward pass (efficiency).
- Saves `generator.pth`, `discriminator.pth`, `generator_ema.pth` every `SAVE_EVERY` epochs, on `Ctrl+C`, and at the end.
- Saves a preview grid (`outputs/epoch_XXX.png`, generated from the EMA weights) every `SAVE_EVERY` epochs.
- Logs per-epoch losses to `outputs/loss_log.csv`.
- Progress bar shows `G`/`D` (losses) and `Dr`/`Df` (Discriminator's average confidence on real/fake batches — useful for spotting an imbalanced Generator/Discriminator fight).

**Usage:**
```bash
python source/train.py
```
No arguments — everything is controlled via `config.py`.

---

## `generate.py`

Generates images from a trained Generator checkpoint.

**Usage / features:**
```bash
# random hair/eyes, uses the EMA generator by default
python source/generate.py

# specific labels
python source/generate.py --hair blonde --eyes blue

# list every label your dataset actually has
python source/generate.py --preview

# use the raw (non-averaged) generator instead of EMA
python source/generate.py --raw

# other options
python source/generate.py --num_images 32 --seed 42 --output my_grid.png --save_individual --checkpoint path/to/custom.pth
```

| Flag | Effect |
|---|---|
| `--hair`, `--eyes` | fix the label instead of sampling randomly |
| `--num_images` | how many images to generate |
| `--seed` | reproducible output |
| `--output` | output filename (saved under `outputs/`) |
| `--checkpoint` | load a specific `.pth` file instead of the default |
| `--preview` | just print the available hair/eye label names, generate nothing |
| `--save_individual` | also save each image separately under `outputs/generated/`, not just the grid |
| `--raw` | use `generator.pth` instead of `generator_ema.pth` |

---

## EMA vs. Raw, in short

- **Raw** = the Generator's exact current weights.
- **EMA** = a running average of those weights over training. Usually smoother/cleaner-looking; can lag behind on very short training runs, which is why `--raw` exists as a comparison point.

## Reading `Dr` / `Df` During Training

- Both sitting mid-range (roughly 0.3–0.7) → healthy, balanced training.
- `Dr` near 1.0 and `Df` near 0.0 for a long stretch → Discriminator is dominating; Generator has little useful signal.

## Typical Expectations

Anime-face GANs at this scale commonly need well past 100 epochs before faces clean up — rough structure (poses, hair, blurry faces) by ~epoch 100 with continued improvement through 200+ is normal, not a sign of failure.