import glob as glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch

mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())
resize_crop = transforms.Compose([transforms.Resize(96),
                                  transforms.CenterCrop(96)])
tensor_normalize = transforms.Compose([transforms.ToTensor(),
                                       normalize])

img_tensors = torch.tensor([])
i = 0
for img_path in glob.glob('./dataset/*.jpg'):
    print(img_path, i)
    i += 1
    img = resize_crop(Image.open(img_path))
    initial_img = tensor_normalize(img)
    rotated_img = tensor_normalize(img.rotate(180))
    v_flipped_img = tensor_normalize(img.transpose(Image.FLIP_TOP_BOTTOM))
    h_flipped_img = tensor_normalize(img.transpose(Image.FLIP_LEFT_RIGHT))
    stacked_imgs = torch.stack([initial_img, rotated_img, v_flipped_img, h_flipped_img])
    img_tensors = torch.cat([img_tensors, stacked_imgs])

print(img_tensors.size())
img_np_arr = img_tensors.numpy()
print(img_np_arr.shape)

np.savez('dataset.npz', train_arr=img_np_arr)