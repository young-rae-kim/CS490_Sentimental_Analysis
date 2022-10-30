import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageClassificationDataset(Dataset):
    def __init__(self, image_files, labels=None, img_transform=None, rgb_only=True):
        super(ImageClassificationDataset, self).__init__()
        self.image_files = image_files
        self.labels = labels
        self.img_transform = img_transform
        self.rgb_only = rgb_only

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])

        if self.rgb_only and img.mode != 'RGB':
            img = img.convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)

        label = []
        if self.labels is not None:
            label = self.labels[index]

        res = {'image': img, 'label': label, 'index': index}
        return res

    def __len__(self):
        return len(self.image_files)