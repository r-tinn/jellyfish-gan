import glob as glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch

mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
resize_crop = transforms.Compose([transforms.Resize(32),
                                  transforms.CenterCrop(32)])
tensor_normalize = transforms.Compose([transforms.ToTensor(),
                                       normalize])
unnormalize_pil = transforms.Compose([unnormalize,
                                      transforms.ToPILImage()])
i = 0
for img_path in glob.glob("./check/*.jpg"):
    img = Image.open(img_path)
    # img = img.convert('1')
    img = resize_crop(img)
    img.save(str(i) + "tmp.jpg")
    # img = tensor_normalize(img)
    # img = unnormalize_pil(img)

    # # img = Image.fromarray(arr)
    # img.save(str(i) + "tmp.jpg")
    i += 1
    # img = transformations(img)
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # img.save("tmp_tb.jpg")
    # img = transformations(img)
    # img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # img.save("tmp_lr.jpg")