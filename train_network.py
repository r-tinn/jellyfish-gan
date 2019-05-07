'''usage: train_network.py dataset_file
Train a DC-GAN on the image dataset.'''
import argparse
from models import JellyfishDataset, Generator, Discriminator, init_weights
from data_preprocess import tensor_unnorm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset_file', type=str)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--num_imgs', type=int, default=10)
opt = parser.parse_args()
print(opt)

# Create output directory for synthesised images
os.makedirs("imgs", exist_ok=True)

# Load dataset
jellyfish_dataset = torch.from_numpy(np.load(opt.dataset_file)['train_arr'])
dataloader = DataLoader(jellyfish_dataset,
                        batch_size=opt.batch_size,
                        shuffle=True)
print("Loaded dataset")

# Initialise generator
generator = Generator(opt.img_size, opt.latent_dim)
generator.conv_layers.apply(init_weights)
generator.linear_layer.apply(init_weights)

# Initialise discriminator
discriminator = Discriminator(opt.img_size)
discriminator.conv_layers.apply(init_weights)
discriminator.linear_layer.apply(init_weights)

# Initialise loss function and optimizers
adversarial_loss = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))

#  Training
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        ground_truth_imgs = Variable(imgs.type(torch.FloatTensor))
        ground_truth_labels = Variable(torch.FloatTensor(
            imgs.shape[0], 1).fill_(1.0),
            requires_grad=False)
        fake_img_labels = Variable(torch.FloatTensor(
            imgs.shape[0], 1).fill_(0.0),
            requires_grad=False)

        # Train generator
        optimizer_G.zero_grad()
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0],
                                       opt.latent_dim))))
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), ground_truth_labels)
        g_loss.backward()
        optimizer_G.step()

        # Train discriminator
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(
            Variable(imgs.type(torch.FloatTensor))),
                               ground_truth_labels)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),
                                     fake_img_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print("Epoch: %d/%d, Batch: %d/%d, D loss: %f, G loss: %f"
              % (epoch,
                 opt.n_epochs,
                 i,
                 len(dataloader),
                 d_loss.item(),
                 g_loss.item()))

        batches_completed = epoch * len(dataloader) + i
        if batches_completed % opt.display_interval == 0:
            for j in range(opt.num_imgs):
                save_image(tensor_unnorm(gen_imgs.data[j]),
                           'imgs/batch-%d_img-%d.png' % (batches_completed, j))
