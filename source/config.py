"""
config.py

Configuration file for the Conditional DCGAN project.
"""

from pathlib import Path
import torch

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# Image Settings
# ==========================================================

IMAGE_SIZE = 64          # Resize all images to 64x64
IMAGE_CHANNELS = 3       # RGB

# ==========================================================
# Dataset
# ==========================================================

NUM_HAIR_CLASSES = 10
NUM_EYE_CLASSES = 6

# ==========================================================
# Model Architecture
# ==========================================================

LATENT_DIM = 100         # Random noise vector size
FEATURE_MAPS = 64        # Base feature maps for Generator/Discriminator

# ==========================================================
# Training
# ==========================================================

BATCH_SIZE = 64
EPOCHS = 100

LEARNING_RATE = 2e-4

BETA1 = 0.5
BETA2 = 0.999

# EMA decay for the shadow generator. Higher = smoother/slower-updating
# average. 0.999 is a standard value used in most modern GAN recipes.
EMA_DECAY = 0.999

# ==========================================================
# Hardware
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 4
PIN_MEMORY = DEVICE.type == "cuda"

# Mixed precision training only makes sense (and is only safe/supported)
# on CUDA.
USE_AMP = DEVICE.type == "cuda"

# cudnn.benchmark speeds up training when input sizes are fixed (they are
# here, since every image is resized to IMAGE_SIZE). Enabled in train.py.
CUDNN_BENCHMARK = DEVICE.type == "cuda"

# ==========================================================
# Random Seed
# ==========================================================

SEED = 42

# ==========================================================
# Checkpoints
# ==========================================================

SAVE_EVERY = 5

GENERATOR_PATH = MODELS_DIR / "generator.pth"
DISCRIMINATOR_PATH = MODELS_DIR / "discriminator.pth"

# EMA (exponential moving average) copy of the generator's weights.
# Produces smoother, generally higher-quality samples than the raw
# generator, since raw GAN weights oscillate step to step during
# adversarial training. Saved alongside the regular checkpoints.
EMA_GENERATOR_PATH = MODELS_DIR / "generator_ema.pth"

# ==========================================================
# Image Generation
# ==========================================================

GENERATED_IMAGES = 64     # Images to generate while monitoring training