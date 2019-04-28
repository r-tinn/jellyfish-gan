import argparse

from old_create_dataset import JellyfishDataset

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--channels', type=int, default=3)
opt = parser.parse_args()
print(opt)
