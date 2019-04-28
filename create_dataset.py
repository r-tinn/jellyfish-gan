import glob as glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--norm_const', type=int, default=0.5)
parser.add_argument('--img_dim', type=int, default=32)
opt = parser.parse_args()
print(opt)

mean = torch.tensor([_NORM_CONST, _NORM_CONST, _NORM_CONST],
                    dtype=torch.float32)
std = torch.tensor([_NORM_CONST, _NORM_CONST, _NORM_CONST],
                   dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())
resize_crop = transforms.Compose([transforms.Resize(_IMG_DIM),
                                  transforms.CenterCrop(_IMG_DIM)])
tensor_norm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean.tolist(),
                                                       std.tolist())])

img_tensors = torch.tensor([])
for img_path in glob.glob('./dataset/*.jpg'):
    img = resize_crop(Image.open(img_path))
    i_stack = torch.stack([tensor_norm(img),
                           tensor_norm(img.rotate(180)),
                           tensor_norm(img.transpose(Image.FLIP_TOP_BOTTOM)),
                           tensor_norm(img.transpose(Image.FLIP_LEFT_RIGHT))])
    img_tensors = torch.cat([img_tensors, i_stack])

np.savez('final_dataset.npz', train_arr=img_tensors.numpy())
