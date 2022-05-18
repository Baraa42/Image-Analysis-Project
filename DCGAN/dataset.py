import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

# build data set
class BAYCDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super(BAYCDataset, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_hash = self.annotations["IPFS HASH"][idx]
        img_path = os.path.join(self.root_dir, img_hash + ".png")
        # img = plt.imread(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(img)
        return img
