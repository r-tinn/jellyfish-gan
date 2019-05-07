import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif type(layer) == nn.BatchNorm2d:
        torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(layer.bias.data, 0.0)


class JellyfishDataset(Dataset):
    def __init__(self, dataset_file):
        self.image_tensors = np.load(datset_file)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx]


class Generator(nn.Module):
    def __init__(self, img_size, z_dim):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.linear_layer = nn.Linear(z_dim, 128*self.init_size**2)
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, z):
        result = self.linear_layer(z)
        return self.conv_layers(result.view(result.shape[0],
                                128,
                                self.init_size,
                                self.init_size))


class Discriminator(nn.Module):
    def __init__(self, img_size,):
        super(Discriminator, self).__init__()
        self.downsampled_img_size = img_size//2**4
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8))

        self.linear_layer = nn.Sequential(
            nn.Linear(128*self.downsampled_img_size**2, 1),
            nn.Sigmoid())

    def forward(self, img):
        result = self.conv_layers(img)
        result = result.view(result.shape[0], -1)
        return self.linear_layer(result)
