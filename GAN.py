import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from datetime import datetime

# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
  def __init__(self, ngpu, nc, nz, ngf):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.nc = nc
    self.nz = nz
    self.ngf = ngf
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf*2) x 14 x 14
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh(),
      # state size. (nc) x 28 x 28
    )

  def forward(self, input):
    return self.main(input)

class Discriminator(nn.Module):
  def __init__(self, ngpu, nc, ndf):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.nc = nc
    self.ndf = ndf
    self.main = nn.Sequential(
      # inputs is (nc) x 28 x 28
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 14 x 14
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 8 x 8
      nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 4 x 4
      nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, input):
    return self.main(input)

class DCGan():
  def __init__(self, dataloader, num_epochs, ngpu, nc, nz, ngf, ndf, lr, beta1):
    self.dataloader = dataloader
    self.num_epochs = num_epochs
    self.ngpu = ngpu
    self.nc = nc
    self.nz = nz
    self.ngf = ngf
    self.ndf = ndf
    self.lr = lr
    self.beta1 = beta1
    self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    # Create the generator
    netG = Generator(ngpu, nc, nz, ngf).to(self.device)
    # Handle multi-gpu if desired
    if (self.device.type == 'cuda') and (ngpu > 1):
      netG = nn.DataParallel(netG, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu, nc, ndf).to(self.device)
    # Handle multi-gpu if desired
    if (self.device.type == 'cuda') and (ngpu > 1):
      netD = nn.DataParallel(netD, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    self.netG = netG
    self.netD = netD

    # Initialize BCELoss function
    self.criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

    # Establish convention for real and fake labels during training
    self.real_label = 1
    self.fake_label = 0

    # Setup Adam optimizers for both G and D
    self.optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    self.optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

  def train(self):
    # Training Loop

    start_datetime = datetime.now()
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(self.num_epochs):
      # For each batch in the dataloader
      for i, data in enumerate(self.dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        self.netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), self.real_label, device=self.device)
        # Forward pass real batch through D
        output = self.netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()

        # Output training stats
        if i % 50 == 0:
          print('[%s][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (datetime.now().strftime("%y/%m/%d %H:%M:%S"),
              epoch, self.num_epochs, i, len(self.dataloader),
              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
          with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
          # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
          vutils.save_image(fake,
            f'output/fake_samples_epoch_{epoch}.png',
            normalize=True)

        iters += 1
      # do checkpointing
      if epoch % 5 == 0:
        torch.save(self.netG, f'output/netG_epoch_{epoch + 1}.pth')
        torch.save(self.netD, f'output/netD_epoch_{epoch + 1}.pth')
    
    finish_datetime = datetime.now()
    deltatime = int((finish_datetime - start_datetime).total_seconds())
    print(f'Started training at finished {start_datetime.strftime("%y/%m/%d %H:%M:%S")} at {finish_datetime.strftime("%y/%m/%d %H:%M:%S")}')
    print('... %0.2fs = %0.2fm = %0.2fh elapsed' % (deltatime, deltatime / 60, deltatime / 3600))
    # Save for plotting
    self.G_losses = G_losses
    self.D_losses = D_losses
    self.img_list = img_list
  
  def plot_D_G_losses(self):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(self.G_losses,label="G")
    plt.plot(self.D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

  def plot_generator_progression(self):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

  def plot_real_vs_fake_images(self):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(self.dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(self.img_list[-1],(1,2,0)))
    plt.show()