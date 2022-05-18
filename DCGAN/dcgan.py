import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import dataset

from model import initialize_weights, Generator, Discriminator
from utils import unnorm, show_batch_images
from PIL import Image


# hyperparameters
# NOTE: according to the paper DCGAN is sensible to hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
REAL_SIZE = 631
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
norm = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)


# Dataset
# Link to get dataset: https://www.kaggle.com/datasets/stanleyjzheng/bored-apes-yacht-club
csv_file = "./bayc/hashes.csv"
root_dir = "./bayc/"
# csv_file = "/content/drive/MyDrive/bayc/hashes.csv"
# root_dir = "/content/drive/MyDrive/bayc/"
dataset = dataset.BAYCDataset(csv_file, root_dir, transform=transform)

# Dataloader
loader = DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True
)


# instantiate both models
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
discriminator = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
# generator = torch.load("./trained-models/generator.pth")
# discriminator = torch.load("./trained-models/discriminator.pth")

# init weights
initialize_weights(generator)
initialize_weights(discriminator)

# optimizer and loss, maybe BCEWithLogits is better here
opt_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

drive_path = "/content/drive/MyDrive/bayc/"  # change by path to save data
save_path = os.path.join(drive_path, "gan_images_bayc")
model_path = os.path.join(drive_path, "models")

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)


run = 1
NUM_EPOCHS = 25
NUMS_TRAINS_GENERATOR = 1
d_losses = np.zeros(NUM_EPOCHS)
g_losses = np.zeros(NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):
    d_loss = []
    g_loss = []

    for idx, inputs in enumerate(loader):
        inputs = inputs.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM)).to(device)

        ### Train Discriminator
        ## get fake images
        fakes = generator(noise)

        outputs_real = discriminator(inputs).view(-1)
        outputs_fake = discriminator(fakes).view(-1)

        loss_real = criterion(outputs_real, torch.ones_like(outputs_real))
        loss_fake = criterion(outputs_fake, torch.zeros_like(outputs_fake))
        loss_d = 0.5 * (loss_real + loss_fake)

        opt_d.zero_grad()
        loss_d.backward(retain_graph=True)
        opt_d.step()
        d_loss.append(loss_d.item())

        ## Train generator more times more than disc
        for _ in range(NUMS_TRAINS_GENERATOR):
            noise = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
            fakes = generator(noise)
            outputs_fake = discriminator(fakes).view(-1)
            loss_g = criterion(outputs_fake, torch.ones_like(outputs_fake))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
        g_loss.append(loss_g.item())

        d_losses[epoch] = np.mean(d_loss)
        g_losses[epoch] = np.mean(g_loss)

        # Print losses occasionally and print to tensorboard
        if idx % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch {idx}/{len(loader)} \
                    Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}"
            )

            show_batch_images(inputs, "Real Image")
            show_batch_images(fakes, "Fake Image")
            save_image(
                unnorm(fakes, *norm),
                os.path.join(save_path, f"{run}_{epoch}_{idx//100}.png"),
            )

torch.save(generator, os.path.join(model_path, "generator.pth"))
torch.save(discriminator, os.path.join(model_path, "discriminator.pth"))
print("MODEL SAVED")

# some plots
plt.plot(d_losses, label="disc_loss")
plt.plot(g_losses, label="gen_loss")
plt.legend()
plt.show()


generator.eval()
discriminator.eval()


def show_sample():
    noise = torch.randn((1, 100)).to(device)
    fake = generator(noise)
    unnorm_images = (
        unnorm(fake, *norm).permute(0, 2, 3, 1).clamp(0, 1).cpu().detach().numpy()
    )
    plt.imshow(unnorm_images[0])
    plt.show()


# show generated samples
while True:
    show_sample()
    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith("n"):
        break
    else:
        plt.cla()
