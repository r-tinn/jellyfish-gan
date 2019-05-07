'''usage: data_preprocess.py input_path output_file
Create a dataset of numpy arrays from a folder of images.'''

import glob as glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str)
parser.add_argument('--norm_const', type=int, default=0.5)
parser.add_argument('--img_dim', type=int, default=32)
opt = parser.parse_args()
print(opt)

mean = torch.tensor([opt.norm_const, opt.norm_const, opt.norm_const],
                    dtype=torch.float32)
std = torch.tensor([opt.norm_const, opt.norm_const, opt.norm_const],
                   dtype=torch.float32)
resize_crop = transforms.Compose([transforms.Resize(opt.img_dim),
                                  transforms.CenterCrop(opt.img_dim)])
tensor_norm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean.tolist(),
                                                       std.tolist())])
tensor_unnorm = transforms.Normalize((-mean / std).tolist(),
                                     (1.0 / std).tolist())

if __name__ == '__main__':
    img_tensors = torch.tensor([])
    for img_path in glob.glob(opt.input_path):
        img = resize_crop(Image.open(img_path))
        i_stack =  \
            torch.stack([tensor_norm(img),
                         tensor_norm(img.rotate(180)),
                         tensor_norm(img.transpose(Image.FLIP_TOP_BOTTOM)),
                         tensor_norm(img.transpose(Image.FLIP_LEFT_RIGHT))])
        img_tensors = torch.cat([img_tensors, i_stack])

    np.savez(opt.output_file, x=img_tensors.numpy())
