
"""
Week 2 - Vanilla GAN on Anime Face Dataset
Author: <Your Name>

A simple, clean implementation of a Vanilla GAN using fully connected
layers. This project is meant for learning how GANs work before moving
to DCGANs and Conditional GANs.
"""

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = "anime_dataset/images"
OUTPUT_DIR = "generated_images"
CHECKPOINT_DIR = "checkpoints"

IMAGE_SIZE = 32
CHANNELS = 3
LATENT_DIM = 100

BATCH_SIZE = 64
EPOCHS = 100

LEARNING_RATE = 0.0002
BETA1 = 0.5

SEED = 42

# ==========================================================
# Small helper so results are reproducible.
# ==========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========================================================
# Custom dataset
# ==========================================================

class AnimeDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.images = [
            file
            for file in os.listdir(root_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = os.path.join(
            self.root_dir,
            self.images[index]
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image


dataset = AnimeDataset(DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# ==========================================================
# Generator
# ==========================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.ReLU(True),

            nn.Linear(
                1024,
                IMAGE_SIZE * IMAGE_SIZE * CHANNELS
            ),

            nn.Tanh()
        )

    def forward(self, noise):

        output = self.network(noise)

        output = output.view(
            -1,
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        return output


# ==========================================================
# Discriminator
# ==========================================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                IMAGE_SIZE * IMAGE_SIZE * CHANNELS,
                1024
            ),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image):

        image = image.view(image.size(0), -1)

        return self.network(image)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_g = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, 0.999)
)

optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, 0.999)
)


# ==========================================================
# Training
# ==========================================================

print(f"\nTraining on: {device}")
print(f"Images found : {len(dataset)}\n")

for epoch in range(EPOCHS):

    generator.train()
    discriminator.train()

    running_g = 0
    running_d = 0

    for i, real_images in enumerate(loader):

        real_images = real_images.to(device)

        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Teach the discriminator what real images look like.
        optimizer_d.zero_grad()

        real_loss = criterion(
            discriminator(real_images),
            real_labels
        )

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=device
        )

        fake_images = generator(noise)

        fake_loss = criterion(
            discriminator(fake_images.detach()),
            fake_labels
        )

        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_d.step()

        # Now train the generator to fool the discriminator.
        optimizer_g.zero_grad()

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=device
        )

        generated = generator(noise)

        g_loss = criterion(
            discriminator(generated),
            real_labels
        )

        g_loss.backward()
        optimizer_g.step()

        running_d += d_loss.item()
        running_g += g_loss.item()

        # Show progress every 20 batches
        if (i + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{i + 1}/{len(loader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

    avg_d = running_d / len(loader)
    avg_g = running_g / len(loader)

    print(
        f"Epoch [{epoch + 1:03}/{EPOCHS}] "
        f"| D Loss: {avg_d:.4f} "
        f"| G Loss: {avg_g:.4f}"
    )

    save_image(
        generated[:25],
        os.path.join(
            OUTPUT_DIR,
            f"epoch_{epoch+1}.png"
        ),
        normalize=True,
        nrow=5
    )

    torch.save(
        generator.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "generator.pth"
        )
    )

    torch.save(
        discriminator.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "discriminator.pth"
        )
    )

print("\nTraining completed successfully!")
