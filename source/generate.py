"""
generate.py

Generate anime faces using a trained Conditional DCGAN.

Examples:
    python source/generate.py
    python source/generate.py --hair blonde --eyes blue
    python source/generate.py --num_images 32 --seed 42
    python source/generate.py --preview
    python source/generate.py --raw          # use generator.pth instead of the EMA weights
"""

import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from config import *
from dataset import AnimeDataset
from models import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate anime faces")
    parser.add_argument("--hair", type=str, default=None)
    parser.add_argument("--eyes", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=GENERATED_IMAGES)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="generated.png")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--save_individual", action="store_true")
    parser.add_argument(
        "--raw", action="store_true",
        help="Use the raw generator.pth instead of the EMA-averaged weights.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    dataset = AnimeDataset()

    if args.preview:
        print("Hair Labels")
        for k, v in dataset.hair_to_idx.items():
            print(f"  {k}")
        print("\nEye Labels")
        for k, v in dataset.eye_to_idx.items():
            print(f"  {k}")
        return

    if args.seed is not None:
        set_seed(args.seed)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif args.raw:
        ckpt = GENERATOR_PATH
    else:
        # EMA weights are the recommended default: they're the running
        # average of the generator over training and typically look
        # cleaner than the raw, still-oscillating weights. Falls back to
        # the raw checkpoint if no EMA file exists yet (e.g. an older run).
        ckpt = EMA_GENERATOR_PATH if EMA_GENERATOR_PATH.exists() else GENERATOR_PATH

    if not ckpt.exists():
        raise FileNotFoundError(f"No generator checkpoint found: {ckpt}")

    print(f"Loading checkpoint: {ckpt}")

    netG = Generator().to(DEVICE)
    netG.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    netG.eval()

    n = args.num_images

    if args.hair is None:
        hair = torch.randint(0, len(dataset.hair_classes), (n,), device=DEVICE)
    else:
        if args.hair not in dataset.hair_to_idx:
            raise ValueError(f"Unknown hair label: {args.hair}")
        hair = torch.full((n,), dataset.hair_to_idx[args.hair], device=DEVICE, dtype=torch.long)

    if args.eyes is None:
        eyes = torch.randint(0, len(dataset.eye_classes), (n,), device=DEVICE)
    else:
        if args.eyes not in dataset.eye_to_idx:
            raise ValueError(f"Unknown eye label: {args.eyes}")
        eyes = torch.full((n,), dataset.eye_to_idx[args.eyes], device=DEVICE, dtype=torch.long)

    noise = torch.randn(n, LATENT_DIM, 1, 1, device=DEVICE)

    with torch.no_grad():
        images = netG(noise, hair, eyes)

    out_path = OUTPUTS_DIR / args.output
    save_image((images + 1) / 2, out_path, nrow=int(n ** 0.5) or 1)

    if args.save_individual:
        folder = OUTPUTS_DIR / "generated"
        folder.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            save_image((img + 1) / 2, folder / f"{i:03d}.png")

    print(f"Saved grid to: {out_path}")


if __name__ == "__main__":
    main()