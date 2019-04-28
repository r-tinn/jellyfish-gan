import glob as glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch


class JellyfishDataset(Dataset):
    def __init__(self, dataset_file):
        self.image_tensors = np.load(datset_file)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx]
