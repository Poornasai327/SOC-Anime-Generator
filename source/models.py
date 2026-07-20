"""
models.py

Production-quality Conditional DCGAN models for anime face generation.

Stability & quality fixes vs. the original version:
  1. Embedding weights are explicitly initialized to a small std (0.02)
     instead of PyTorch's default N(0, 1). Previously, the discriminator's
     label-conditioning channels had much larger magnitude than the real
     image channels (which live in [-1, 1]), letting noisy label signal
     dominate the first conv layer.
  2. Discriminator condition maps are passed through tanh so they're
     bounded to [-1, 1] -- the same range as the real image channels
     they're concatenated with.
  3. Discriminator uses spectral normalization instead of BatchNorm.
     BatchNorm + long adversarial training is a common cause of the
     discriminator overpowering the generator (vanishing G gradients).
  4. Discriminator returns raw logits (no final Sigmoid), paired with
     BCEWithLogitsLoss in train.py for numerical stability.
  5. FIX: weights_init now correctly initializes spectral-norm-wrapped
     conv layers. spectral_norm() reparameterizes `weight` into
     `weight_orig` + a value recomputed on every forward pass -- directly
     initializing `.weight` on such a layer is silently discarded on the
     first forward. weights_init now targets `weight_orig` when present.
  6. Added a lightweight self-attention block (SAGAN-style) at the 16x16
     feature-map stage in both G and D, to help capture longer-range
     structure (eye/hair symmetry, coherent strand direction) that plain
     local convolutions miss.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from config import (
    LATENT_DIM,
    FEATURE_MAPS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    NUM_HAIR_CLASSES,
    NUM_EYE_CLASSES,
)

EMBED_DIM = 50
CONDITION_DIM = EMBED_DIM * 2


def weights_init(module: nn.Module) -> None:
    """Initialize model weights following DCGAN recommendations."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        # FIX: spectral_norm renames the learnable parameter to
        # `weight_orig`; `weight` becomes a derived, non-parameter
        # attribute recomputed each forward. Initialize whichever
        # actually holds the learnable data.
        target = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(target, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        # FIX: default init is N(0, 1), much larger in magnitude than the
        # [-1, 1]-normalized image data these embeddings sit alongside.
        nn.init.normal_(module.weight, 0.0, 0.02)


class SelfAttention(nn.Module):
    """
    Lightweight SAGAN-style self-attention block. Lets every spatial
    position attend to every other position, helping the network keep
    things like eye symmetry and hair-strand direction consistent across
    the whole face rather than only within a local receptive field.

    gamma starts at 0, so the block is a no-op at initialization and the
    model learns how much to rely on attention over training -- this
    keeps it from destabilizing early training.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        q = self.query(x).view(b, -1, n).permute(0, 2, 1)  # B, N, C//8
        k = self.key(x).view(b, -1, n)                      # B, C//8, N
        attn = torch.softmax(torch.bmm(q, k), dim=-1)        # B, N, N

        v = self.value(x).view(b, -1, n)                     # B, C, N
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)

        return x + self.gamma * out


class GeneratorBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int, padding: int):
        super().__init__(
            nn.ConvTranspose2d(
                in_channels, out_channels, 4, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )


class DiscriminatorBlock(nn.Sequential):
    """Spectral norm instead of BatchNorm2d -- see module docstring."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )


class Generator(nn.Module):
    """
    Conditional DCGAN Generator.

    Spatial progression for IMAGE_SIZE=64:
        input 1x1 -> block1 4x4 -> block2 8x8 -> block3 16x16
        -> [self-attention] -> block4 32x32 -> out_conv 64x64
    """

    def __init__(self):
        super().__init__()

        self.hair_embed = nn.Embedding(NUM_HAIR_CLASSES, EMBED_DIM)
        self.eye_embed = nn.Embedding(NUM_EYE_CLASSES, EMBED_DIM)

        self.block1 = GeneratorBlock(LATENT_DIM + CONDITION_DIM, FEATURE_MAPS * 8, 1, 0)
        self.block2 = GeneratorBlock(FEATURE_MAPS * 8, FEATURE_MAPS * 4, 2, 1)
        self.block3 = GeneratorBlock(FEATURE_MAPS * 4, FEATURE_MAPS * 2, 2, 1)
        self.attn = SelfAttention(FEATURE_MAPS * 2)
        self.block4 = GeneratorBlock(FEATURE_MAPS * 2, FEATURE_MAPS, 2, 1)

        self.out_conv = nn.ConvTranspose2d(
            FEATURE_MAPS, IMAGE_CHANNELS, 4, 2, 1, bias=False
        )
        self.out_act = nn.Tanh()

    def create_condition(self, hair, eyes):
        cond = torch.cat(
            [self.hair_embed(hair), self.eye_embed(eyes)], dim=1
        )
        return cond.unsqueeze(-1).unsqueeze(-1)

    def forward(self, noise, hair, eyes):
        x = torch.cat([noise, self.create_condition(hair, eyes)], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attn(x)
        x = self.block4(x)
        x = self.out_conv(x)
        return self.out_act(x)


class Discriminator(nn.Module):
    """
    Conditional DCGAN Discriminator.

    NOTE: forward() returns raw logits, NOT probabilities. Pair with
    BCEWithLogitsLoss, not BCELoss.

    Spatial progression for IMAGE_SIZE=64:
        input 64x64 -> in_conv 32x32 -> block1 16x16
        -> [self-attention] -> block2 8x8 -> block3 4x4 -> out_conv 1x1
    """

    def __init__(self):
        super().__init__()

        pixels = IMAGE_SIZE * IMAGE_SIZE
        self.hair_embed = nn.Embedding(NUM_HAIR_CLASSES, pixels)
        self.eye_embed = nn.Embedding(NUM_EYE_CLASSES, pixels)

        self.in_conv = spectral_norm(
            nn.Conv2d(IMAGE_CHANNELS + 2, FEATURE_MAPS, 4, 2, 1, bias=False)
        )
        self.in_act = nn.LeakyReLU(0.2, inplace=True)

        self.block1 = DiscriminatorBlock(FEATURE_MAPS, FEATURE_MAPS * 2)
        self.attn = SelfAttention(FEATURE_MAPS * 2)
        self.block2 = DiscriminatorBlock(FEATURE_MAPS * 2, FEATURE_MAPS * 4)
        self.block3 = DiscriminatorBlock(FEATURE_MAPS * 4, FEATURE_MAPS * 8)

        self.out_conv = spectral_norm(
            nn.Conv2d(FEATURE_MAPS * 8, 1, 4, 1, 0, bias=False)
        )

    def create_condition_map(self, hair, eyes):
        b = hair.size(0)
        h = torch.tanh(self.hair_embed(hair)).view(b, 1, IMAGE_SIZE, IMAGE_SIZE)
        e = torch.tanh(self.eye_embed(eyes)).view(b, 1, IMAGE_SIZE, IMAGE_SIZE)
        return torch.cat([h, e], dim=1)

    def forward(self, images, hair, eyes):
        x = torch.cat(
            [images, self.create_condition_map(hair, eyes)], dim=1
        )
        x = self.in_act(self.in_conv(x))
        x = self.block1(x)
        x = self.attn(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.out_conv(x)
        return x.view(-1)  # raw logits


if __name__ == "__main__":
    G = Generator()
    D = Discriminator()

    G.apply(weights_init)
    D.apply(weights_init)

    batch_size = 8
    z = torch.randn(batch_size, LATENT_DIM, 1, 1)
    hair = torch.randint(0, NUM_HAIR_CLASSES, (batch_size,))
    eyes = torch.randint(0, NUM_EYE_CLASSES, (batch_size,))

    fake = G(z, hair, eyes)
    logits = D(fake, hair, eyes)

    assert fake.shape == (batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    assert logits.shape == (batch_size,)
    assert torch.isfinite(fake).all()
    assert torch.isfinite(logits).all()

    print("=" * 50)
    print("Generator Output     :", fake.shape)
    print("Discriminator Logits :", logits.shape)
    print("Sample logit range   :", logits.min().item(), "to", logits.max().item())
    print("All tests passed.")
    print("=" * 50)