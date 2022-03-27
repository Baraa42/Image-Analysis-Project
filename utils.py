import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# scale back image
def scale_image(img):
    out = (img + 1) /2
    return out

# to plot during training
def plot_images(images, batch_size=128, cols=5, rows=5):
    fig = plt.figure(figsize=(8, 8))
    imgs = images.cpu().detach().reshape(-1, 64, 64)
    for i in range(1, cols*rows +1):
        idx = np.random.randint(batch_size)
        img = imgs[i]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)#, cmap='gray')
    plt.show()
    

def unnorm(images, means, stds):
    means = torch.tensor(means).reshape(1,3,1,1).to(device)
    stds = torch.tensor(stds).reshape(1,3,1,1).to(device)
    return images*stds+means