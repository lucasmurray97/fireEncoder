import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.name = "VAE"
        self.instance= params["instance"]
        self.latent_dims = params["latent_dims"]
        self.c = params["capacity"]
        self.input_size = params["input_size"]
        kernel_size = 4
        stride = 2
        padding = 1
        self.dim_1 = int((self.input_size - kernel_size + 2*padding)/2 + 1)
        self.dim_2 = int((self.dim_1 - kernel_size + 2*padding)/2 + 1)
        sigmoid = params["sigmoid"]
        self.is_sigmoid = sigmoid
        self.lr1 = params["lr1"]
        self.not_reduced = params["not_reduced"]
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # By default set to training
        self.training =  True
        # Encoder layers:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding) # (64, 10, 10)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=kernel_size, stride=stride, padding=padding) # (128, 5, 5)
        self.fc_mu = nn.Linear(in_features=self.latent_dims*(self.dim_2**2), out_features = self.latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.latent_dims*(self.dim_2**2), out_features = self.latent_dims)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.drop1 = nn.Dropout()
        self.bn2 = nn.BatchNorm2d(self.c*2)
        self.drop2 = nn.Dropout()
        # Decoder layers:
        self.fc = nn.Linear(in_features=self.latent_dims, out_features=self.latent_dims*(self.dim_2**2))
        self.conv1_ = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_ = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.criterion = nn.BCELoss() if self.is_sigmoid else nn.MSELoss()
        self.bn1_2 = nn.BatchNorm2d(self.c)
        self.drop1_2 = nn.Dropout()
        # Inicialización de parámetros:
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2_.weight, mode='fan_in', nonlinearity='relu')
        # Optimizer:
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr1)
        # Losses holders:
        self.training_loss = []
        self.validation_loss = []
        self.reconstruction_training_loss = []
        self.reconstruction_validation_loss = []
        self.divergence_training_loss = []
        self.divergence_validation_loss = []
        self.m = 0
        self.n = 0
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = 0
        self.val_reconstruction_epoch_loss = 0
        self.divergence_epoch_loss = 0
        self.val_divergence_epoch_loss = 0
        self.variational_beta = params["variational_beta"]
        self.last_mu = None
        self.last_logvar = None
        print(self.variational_beta)
        
    def encode(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop1(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.drop2(F.relu(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

    def decode(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*2, self.dim_2, self.dim_2) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = self.bn1_2(self.conv1_(x))
        x = self.drop1_2(F.relu(x))
        x = torch.sigmoid(self.conv2_(x)) if self.is_sigmoid else F.relu(self.conv2_(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
    def forward(self, x, r):
        latent_mu, latent_logvar = self.encode(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decode(latent)
        self.last_mu = latent_mu
        self.last_logvar = latent_logvar
        return x_recon
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            # print(std, eps)
            sample = eps.mul(std).add_(mu)
            return sample
        else:
            return mu
    
    def vae_loss(self, recon_x, x, mu, logvar):
        # recon_x is the probability of a multivariate Bernoulli distribution p.
        # -log(p(x)) is then the pixel-wise binary cross-entropy.
        # Averaging or not averaging the binary cross-entropy over all pixels here
        # is a subtle detail with big effect on training, since it changes the weight
        # we need to pick for the other loss term by several orders of magnitude.
        # Not averaging is the direct implementation of the negative log likelihood,
        # but averaging makes the weight of the other loss term independent of the image resolution.
        if self.is_sigmoid:
            if not self.not_reduced:
                recon_loss = F.binary_cross_entropy(recon_x.view(-1, 400), x.view(-1, 400), reduction='sum')
            else:
                recon_loss = self.criterion(recon_x, x)
        else:
            recon_loss = self.criterion(recon_x, x)
        # KL-divergence between the prior distribution over latent vectors
        # (the one we are going to sample from when generating new images)
        # and the distribution estimated by the generator for the given image.
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kldivergence, recon_loss + self.variational_beta * kldivergence

    def loss(self, output, x, r):
        mu, logvar = (self.last_mu, self.last_logvar)
        loss_1, loss_2, loss = self.vae_loss(output, x, mu, logvar)
        self.reconstruction_epoch_loss += loss_1.item()
        self.divergence_epoch_loss += loss_2.item()
        self.epoch_loss += loss.item()
        return loss

    
    def val_loss(self, output, x, r):
        mu, logvar = (self.last_mu, self.last_logvar)
        loss_1, loss_2, loss = self.vae_loss(output, x, mu, logvar)
        self.val_reconstruction_epoch_loss += loss_1.item()
        self.val_divergence_epoch_loss += loss_2.item()
        self.val_epoch_loss += loss.item()
        return loss
        
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def reset_losses(self):
        self.training_loss.append(self.epoch_loss/self.n)
        self.validation_loss.append(self.val_epoch_loss/self.m)
        self.reconstruction_training_loss.append(self.reconstruction_epoch_loss/self.n)
        self.reconstruction_validation_loss.append(self.val_reconstruction_epoch_loss/self.m)
        self.divergence_training_loss.append(self.divergence_epoch_loss/self.n)
        self.divergence_validation_loss.append(self.val_divergence_epoch_loss/self.m)
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = 0
        self.val_reconstruction_epoch_loss = 0
        self.divergence_epoch_loss = 0
        self.val_divergence_epoch_loss = 0
        self.m = 0
        self.n = 0
    
    def plot_loss(self, epochs):
        self.to("cpu")
        plt.ion()
        fig = plt.figure()
        plt.plot(self.training_loss[1:], label='training loss')
        plt.plot(self.validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_lr1={self.lr1}_not_reduced={self.not_reduced}_variational_beta={self.variational_beta}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.reconstruction_training_loss[1:], label='training loss')
        plt.plot(self.reconstruction_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/reconstruction_loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_lr1={self.lr1}_not_reduced={self.not_reduced}_variational_beta={self.variational_beta}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.divergence_training_loss[1:], label='training loss')
        plt.plot(self.divergence_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/divergence_loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_lr1={self.lr1}_not_reduced={self.not_reduced}_variational_beta={self.variational_beta}.png")

    def calc_test_loss(self, output, images, r):
        return self.loss(output, images, r)