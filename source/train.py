"""
train.py

Production-quality training script for Conditional DCGAN.
Run:
    python source/train.py

Efficiency / quality changes vs. the original version:
  - Mixed precision (torch.cuda.amp) on CUDA -- meaningful speedup with
    negligible quality cost.
  - cudnn.benchmark enabled -- free speedup since input size is fixed.
  - Real and fake batches are concatenated into a single Discriminator
    forward pass instead of two.
  - DiffAugment (differentiable random translation + brightness/contrast
    jitter), applied identically to real and fake images before they
    reach D. Helps prevent D from overfitting on a modest-sized dataset,
    which otherwise stalls training exactly like you saw at 70 epochs.
  - EMA (exponential moving average) of the generator's weights, saved
    separately as generator_ema.pth. Produces smoother, generally
    higher-quality samples than the raw generator, and is used for the
    periodic preview grids during training.
  - Label smoothing (real target 0.9 instead of 1.0).
  - BCEWithLogitsLoss (required, since Discriminator no longer applies
    Sigmoid internally).
"""

import copy
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from config import *
from dataset import AnimeDataset
from models import Generator, Discriminator, weights_init


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """
    Maintains a shadow copy of the generator whose weights are an
    exponential moving average of the live generator's weights:
        shadow = decay * shadow + (1 - decay) * live

    Raw GAN weights oscillate step to step; the averaged copy is
    noticeably more stable and typically produces cleaner samples.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # FIX: ramp decay up from near 0 toward the target instead of using
        # a fixed high decay from step 1. A fixed decay=0.999 means the
        # shadow needs thousands of steps to "forget" its near-random
        # starting weights -- on a dataset with fewer steps/epoch, the EMA
        # copy can end up looking *worse* than the raw generator simply
        # because it's still averaging in a lot of early garbage. This
        # ramp (standard in BigGAN/StyleGAN) self-adjusts regardless of
        # dataset size: decay stays low early (fast-tracking) and
        # saturates at self.decay once enough steps have passed.
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Minimal differentiable augmentation (translation + brightness +
    contrast), applied identically to real and fake batches before they
    reach the Discriminator. All ops are differentiable, so gradients
    still flow back into the Generator through the fake-image path.

    This is a simplified version of DiffAugment (Zhao et al., 2020) --
    it meaningfully reduces D overfitting on small/medium datasets like
    typical anime-face collections.
    """
    b, c, h, w = x.shape

    # --- random translation (reflect-pad then crop) ---
    shift_h = h // 8
    shift_w = w // 8
    x_padded = F.pad(x, [shift_w, shift_w, shift_h, shift_h], mode="reflect")

    out = torch.empty_like(x)
    ty = torch.randint(0, 2 * shift_h + 1, (b,), device=x.device)
    tx = torch.randint(0, 2 * shift_w + 1, (b,), device=x.device)
    for i in range(b):
        out[i] = x_padded[i, :, ty[i]:ty[i] + h, tx[i]:tx[i] + w]
    x = out

    # --- random brightness ---
    brightness = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    x = x + brightness

    # --- random contrast ---
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    contrast = torch.rand(b, 1, 1, 1, device=x.device) * 0.5 + 0.75
    x = (x - x_mean) * contrast + x_mean

    return x.clamp(-1, 1)


def save_checkpoint(netG, netD, ema):
    torch.save(netG.state_dict(), GENERATOR_PATH)
    torch.save(netD.state_dict(), DISCRIMINATOR_PATH)
    torch.save(ema.shadow.state_dict(), EMA_GENERATOR_PATH)


def save_training_config(dataset_size: int):
    cfg = OUTPUTS_DIR / "training_config.txt"
    with open(cfg, "w") as f:
        f.write(f"Dataset Size : {dataset_size}\n")
        f.write(f"Image Size   : {IMAGE_SIZE}\n")
        f.write(f"Batch Size   : {BATCH_SIZE}\n")
        f.write(f"Epochs       : {EPOCHS}\n")
        f.write(f"LR           : {LEARNING_RATE}\n")
        f.write(f"Latent Dim   : {LATENT_DIM}\n")
        f.write(f"AMP Enabled  : {USE_AMP}\n")
        f.write(f"EMA Decay    : {EMA_DECAY}\n")


def main():
    set_seed(SEED)

    if CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    dataset = AnimeDataset()
    print(f"Training Images : {len(dataset)}")

    # Sanity check: config's class counts must be able to hold every
    # label actually present in the CSV, or the Embedding layers will
    # throw an index-out-of-range error.
    assert len(dataset.hair_classes) <= NUM_HAIR_CLASSES, (
        f"labels.csv has {len(dataset.hair_classes)} hair classes but "
        f"config.NUM_HAIR_CLASSES is {NUM_HAIR_CLASSES}"
    )
    assert len(dataset.eye_classes) <= NUM_EYE_CLASSES, (
        f"labels.csv has {len(dataset.eye_classes)} eye classes but "
        f"config.NUM_EYE_CLASSES is {NUM_EYE_CLASSES}"
    )

    save_training_config(len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )

    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)

    if GENERATOR_PATH.exists():
        print("Loading Generator checkpoint...")
        netG.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    else:
        netG.apply(weights_init)

    if DISCRIMINATOR_PATH.exists():
        print("Loading Discriminator checkpoint...")
        netD.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location=DEVICE))
    else:
        netD.apply(weights_init)

    ema = EMA(netG, decay=EMA_DECAY)
    if EMA_GENERATOR_PATH.exists():
        print("Loading EMA generator checkpoint...")
        ema.shadow.load_state_dict(torch.load(EMA_GENERATOR_PATH, map_location=DEVICE))

    # Discriminator now returns raw logits (no Sigmoid) -- must pair with
    # BCEWithLogitsLoss, not BCELoss.
    criterion = nn.BCEWithLogitsLoss()

    optG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    scalerG = GradScaler(device="cuda", enabled=USE_AMP)
    scalerD = GradScaler(device="cuda", enabled=USE_AMP)

    fixed_noise = torch.randn(GENERATED_IMAGES, LATENT_DIM, 1, 1, device=DEVICE)
    fixed_hair = torch.randint(0, NUM_HAIR_CLASSES, (GENERATED_IMAGES,), device=DEVICE)
    fixed_eyes = torch.randint(0, NUM_EYE_CLASSES, (GENERATED_IMAGES,), device=DEVICE)

    csv_file = OUTPUTS_DIR / "loss_log.csv"
    write_header = not csv_file.exists()

    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "generator_loss", "discriminator_loss"])

            for epoch in range(1, EPOCHS + 1):
                g_epoch = 0.0
                d_epoch = 0.0

                pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

                for images, hair, eyes in pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    hair = hair.to(DEVICE, non_blocking=True)
                    eyes = eyes.to(DEVICE, non_blocking=True)

                    b = images.size(0)
                    # Label smoothing: keeps D from getting overconfident,
                    # which helps avoid vanishing generator gradients.
                    real_labels = torch.full((b,), 0.9, device=DEVICE)
                    fake_labels = torch.zeros(b, device=DEVICE)

                    # ---------------- Discriminator step ----------------
                    optD.zero_grad(set_to_none=True)

                    with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                        noise = torch.randn(b, LATENT_DIM, 1, 1, device=DEVICE)
                        fake_images = netG(noise, hair, eyes)

                        real_aug = diff_augment(images)
                        fake_aug = diff_augment(fake_images.detach())

                        combined_images = torch.cat([real_aug, fake_aug], dim=0)
                        combined_hair = torch.cat([hair, hair], dim=0)
                        combined_eyes = torch.cat([eyes, eyes], dim=0)

                        combined_out = netD(combined_images, combined_hair, combined_eyes)
                        out_real, out_fake = combined_out.chunk(2, dim=0)

                        loss_real = criterion(out_real, real_labels)
                        loss_fake = criterion(out_fake, fake_labels)
                        d_loss = loss_real + loss_fake

                    scalerD.scale(d_loss).backward()
                    scalerD.step(optD)
                    scalerD.update()

                    # ---------------- Generator step ----------------
                    optG.zero_grad(set_to_none=True)

                    with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                        fake_aug_for_g = diff_augment(fake_images)
                        out = netD(fake_aug_for_g, hair, eyes)
                        g_loss = criterion(out, real_labels)

                    scalerG.scale(g_loss).backward()
                    scalerG.step(optG)
                    scalerG.update()

                    ema.update(netG)

                    g_epoch += g_loss.item()
                    d_epoch += d_loss.item()

                    with torch.no_grad():
                        d_real_prob = torch.sigmoid(out_real).mean().item()
                        d_fake_prob = torch.sigmoid(out_fake).mean().item()

                    pbar.set_postfix(
                        G=f"{g_loss.item():.3f}",
                        D=f"{d_loss.item():.3f}",
                        Dr=f"{d_real_prob:.2f}",
                        Df=f"{d_fake_prob:.2f}",
                    )

                avg_g = g_epoch / len(loader)
                avg_d = d_epoch / len(loader)

                writer.writerow([epoch, avg_g, avg_d])
                f.flush()

                print("\n" + "=" * 50)
                print(f"Epoch {epoch}/{EPOCHS}")
                print(f"Generator Loss     : {avg_g:.4f}")
                print(f"Discriminator Loss : {avg_d:.4f}")

                if epoch % SAVE_EVERY == 0:
                    save_checkpoint(netG, netD, ema)
                    with torch.no_grad():
                        # Preview samples come from the EMA generator --
                        # smoother and more representative of final quality
                        # than the raw, still-oscillating live generator.
                        samples = ema.shadow(fixed_noise, fixed_hair, fixed_eyes)
                        save_image(
                            (samples + 1) / 2,
                            OUTPUTS_DIR / f"epoch_{epoch:03d}.png",
                            nrow=8,
                        )
                    print("Checkpoint saved.")
                print("=" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(netG, netD, ema)

    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        print("Saving checkpoint before exiting...")
        save_checkpoint(netG, netD, ema)
        raise

    save_checkpoint(netG, netD, ema)
    print("Training complete.")


if __name__ == "__main__":
    main()