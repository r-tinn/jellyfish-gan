import glob as glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch

class JellyfishDataset(Dataset):

    def __init__(self, dataset_path):
        self.datset_path = dataset_path
        self.transformations = self.create_transformations()
        self.image_tensors = self.images_to_tensors()

    def images_to_tensors(self):
        img_tensors = torch.tensor([])
        for img_path in glob.glob(self.datset_path):
            print(img_path)
            img = Image.open(img_path)
            initial_img = self.transformations(img)
            rotated_img = self.transformations(img.rotate(180))
            v_flipped_img = self.transformations(img.transpose(Image.FLIP_TOP_BOTTOM))
            h_flipped_img = self.transformations(img.transpose(Image.FLIP_LEFT_RIGHT))
            stacked_imgs = torch.stack([initial_img, rotated_img, v_flipped_img, h_flipped_img])
            img_tensors = torch.cat([img_tensors, stacked_imgs])
        return img_tensors

    def create_transformations(self):
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        normalize = transforms.Normalize(mean.tolist(), std.tolist())
        return transforms.Compose([transforms.Resize(96),
                                   transforms.CenterCrop(96),
                                   transforms.ToTensor(),
                                   normalize])
        
    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx]