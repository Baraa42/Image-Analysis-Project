import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import make_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# scale back image
def scale_image(img):
    out = (img + 1) / 2
    return out


# to plot during training
def plot_images(images, batch_size=128, cols=5, rows=5):
    fig = plt.figure(figsize=(8, 8))
    imgs = images.cpu().detach().reshape(-1, 64, 64)
    for i in range(1, cols * rows + 1):
        idx = np.random.randint(batch_size)
        img = imgs[i]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)  # , cmap='gray')
    plt.show()


def unnorm(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1).to(device)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1).to(device)
    return images * stds + means


def show_batch(data_loader):
    for images in data_loader:
        images = images.to(device)
        batch_size = len(images)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_xticks([])
        ax.set_yticks([])
        unnorm_images = unnorm(images, *norm).cpu()
        ax.imshow(
            make_grid(unnorm_images[:batch_size], nrow=8).permute(1, 2, 0).clamp(0, 1)
        )
        break


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def show_batch_images(images, title=""):
    images = images.to(device)
    batch_size = len(images)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    unnorm_images = unnorm(images, *norm).cpu()
    ax.imshow(
        make_grid(unnorm_images[:batch_size], nrow=8).permute(1, 2, 0).clamp(0, 1)
    )
    plt.show()
