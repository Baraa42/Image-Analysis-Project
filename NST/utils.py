import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from PIL import Image

import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


csv_file_path = "/content/drive/MyDrive/bayc/hashes.csv"
root_dir = "/content/drive/MyDrive/bayc/"

img_size = 356
loader = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
)

bayc = dataset.BAYCDataset(csv_file_path, root_dir, loader)


def load_image(image_name):
    img = Image.open(image_name).convert("RGB")
    img = loader(img).unsqueeze(0)
    return img.to(device)


def load_image(image_name=None, idx=None):
    if idx:
        img = bayc[idx]
        img = img.unsqueeze(0)
    elif image_name:
        img = Image.open(image_name).convert("RGB")
        img = loader(img).unsqueeze(0)
    else:
        raise ("Not allowed")

    return img.to(device)


def show_images(images, title=""):
    # images = images.to(device)
    total_imgs = len(images)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    images = [image[0].cpu() for image in images]
    # unnorm_images = unnorm(images, *norm).cpu()
    ax.imshow(
        make_grid(images[:total_imgs], nrow=total_imgs).permute(1, 2, 0)
    )  # .clamp(0,1))
    plt.show()
    # save_image(unnorm_images, f"gan_images/test.png")
