import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from GAN import DCGan

# Root directory for dataset
dataroot = "data/mnist/"
# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 1 # for mnist
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 6
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Create the dataset
dataset = dset.MNIST(
  root=dataroot,
  download=True,
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size=batch_size,
  shuffle=True,
  num_workers=workers
)

# Plot some training images (for test purposes)
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# real_batch = next(iter(dataloader))
# print(real_batch[0].mean(), real_batch[0].min(), real_batch[0].max())
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

# Run the program
Gan = DCGan(dataloader, num_epochs, ngpu, nc, nz, ngf, ndf, lr, beta1)
Gan.train()
Gan.plot_D_G_losses()
#Gan.plot_generator_progression()
#Gan.plot_real_vs_fake_images()