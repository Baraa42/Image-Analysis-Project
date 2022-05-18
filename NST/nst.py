import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image, make_grid


import matplotlib.pyplot as plt
from PIL import Image

import dataset
import model
from utils import load_image, show_images


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 356
csv_file_path = "/content/drive/MyDrive/bayc/hashes.csv"
root_dir = "/content/drive/MyDrive/bayc/"
loader = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
)

bayc = dataset.BAYCDataset(csv_file_path, root_dir, loader)

style_imgs_path = "drive/MyDrive/style-art-images/"
style_imgs = [
    "dali1.jpg",
    "picasso1.png",
    "schrei.jpeg",
    "sw.jpeg",
    "kadinski.jpeg",
    "picasso.jpeg",
    "style1.jpeg",
    "vg1.jpeg",
]

idx = np.random.randint(len(bayc))
style_idx = np.random.randint(len(os.listdir(style_imgs_path)))

style_img_path = os.path.join(style_imgs_path, style_imgs[style_idx])
original_img = load_image(idx=idx)
style_img = load_image(image_name=style_img_path)


cop = style_img.clone()
generated = original_img.clone().requires_grad_(True)
# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
total_steps = 2000
learning_rate = 0.001
alpha = 1
beta = 0.025
model = model.VGG().to(device).eval()
optimizer = optim.Adam([generated], lr=learning_rate)

drive_save_path = "/content/drive/MyDrive/NST-Art"
if not os.path.exists(drive_save_path):
    os.mkdir(drive_save_path)


# run the model on all style imgs
for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = content_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        optimizer.zero_grad()
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - style_feature) ** 2)

        # compute the Gram Matrix
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * content_loss + beta * style_loss

        torch.autograd.set_detect_anomaly(True)
        # total_loss.backward()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    if step % 200 == 0 and step > 0:
        print(f"Loss after {step} steps:", total_loss.item())

        save_image(
            generated,
            os.path.join(drive_save_path, f"generated_{idx}_{style_idx}_{step+1}.png"),
        )
        # generated_copy = generated.clone()
        # image_generated = np.moveaxis(generated_copy[0].detach().cpu().numpy(),0,-1)
        plt.axis("off")
        show_images([style_img, original_img, generated], title=f"Step:{step}")
        # plt.imshow(image_generated)
        plt.show()
