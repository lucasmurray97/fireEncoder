import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class CCVAE(nn.Module):
    def __init__(self, params):
        super(CCVAE, self).__init__()
        self.name = "CCVAE"
        self.instance= params["instance"]
        self.latent_dims = params["latent_dims"]
        self.c = params["capacity"]
        self.input_size = params["input_size"]
        self.distribution_std = params["distribution_std"]
        kernel_size = 4
        stride = 2
        padding = 1
        self.dim_1 = int((self.input_size - kernel_size + 2*padding)/2 + 1)
        self.dim_2 = int((self.dim_1 - kernel_size + 2*padding)/2 + 1)
        sigmoid = params["sigmoid"]
        self.is_sigmoid = sigmoid
        self.lr1 = params["lr1"]
        self.lr2 = params["lr2"]
        self.not_reduced = params["not_reduced"]
        self.variational_beta = params["variational_beta"]
        use_gpu = params["use_gpu"]
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if use_gpu else 'cpu'
        self.latent_portion = params["latent_portion"]
        self.rec_dim = int(self.latent_dims * (1 - self.latent_portion))
        self.burn_dim = int(self.latent_dims * self.latent_portion)
        self.alpha = params["alpha"]
        # By default set to training
        self.training =  True
        # Encoder layers:
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding) # (64, 10, 10)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=kernel_size, stride=stride, padding=padding) # (128, 5, 5)
        self.fc_mu = nn.Linear(in_features=self.latent_dims*(self.dim_2**2), out_features = self.latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.latent_dims*(self.dim_2**2), out_features = self.latent_dims)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.drop1 = nn.Dropout()
        self.bn2 = nn.BatchNorm2d(self.c*2)
        self.drop2 = nn.Dropout()

        # Group all layers of the encoder in a module
        self.encoder = nn.Sequential(
            self.conv1,
            self.bn1,
            self.drop1,
            nn.ReLU(),
            self.conv2,
            self.bn2,
            self.drop2,
            nn.ReLU(),
            nn.Flatten()
        )

        # Group encoder params
        self.encoder_params = list(self.encoder.parameters()) + list(self.fc_mu.parameters()) + list(self.fc_logvar.parameters())

        # Decoder layers:
        self.fc = nn.Linear(in_features=self.latent_dims, out_features=self.latent_dims*(self.dim_2**2))
        self.conv1_ = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_ = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1_2 = nn.BatchNorm2d(self.c)
        self.drop1_2 = nn.Dropout()

        # Group all layers of the decoder in a module
        self.decoder = nn.Sequential(
            self.fc,
            nn.ReLU(),
            nn.Unflatten(1, (self.c*2, self.dim_2, self.dim_2)),
            self.conv1_,
            self.bn1_2,
            self.drop1_2,
            nn.ReLU(),
            self.conv2_
        )
   

        # Burned % predictor:
        self.fc_r1 = nn.Linear(in_features=self.burn_dim, out_features=128)
        self.bn_r1 = nn.BatchNorm1d(128)
        self.fc_r2 = nn.Linear(in_features=128, out_features=64)
        self.bn_r2 = nn.BatchNorm1d(64)
        self.fc_r4 = nn.Linear(in_features=64, out_features=1)
        
        # Group all layers of the burned % predictor in a module
        self.burned_predictor = nn.Sequential(
            self.fc_r1,
            self.bn_r1,
            nn.ReLU(),
            self.fc_r2,
            self.bn_r2,
            nn.ReLU(),
            self.fc_r4
        )
        # Inicialización de parámetros:
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2_.weight, mode='fan_in', nonlinearity='relu')

        # Divide parameters for the vae and the regression task
        self.vae_params = list(self.encoder_params) + list(self.decoder.parameters())
        self.r_params = list(self.burned_predictor.parameters())

        # Optimizer:
        self.optimizer_1 = torch.optim.Adam(self.vae_params, lr = self.lr1)
        self.optimizer_2 = torch.optim.Adam(self.r_params, lr = self.lr2)
        # Loss function:
        self.criterion_vae = nn.BCELoss() if self.is_sigmoid else nn.MSELoss()
        self.criterion_r = nn.MSELoss()
        # Losses holders:
        self.training_loss = []
        self.validation_loss = []
        self.reconstruction_training_loss = []
        self.reconstruction_validation_loss = []
        self.divergence_training_loss = []
        self.divergence_validation_loss = []
        self.burned_training_loss = []
        self.burned_validation_loss = []
        self.m = 0
        self.n = 0
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = 0
        self.val_reconstruction_epoch_loss = 0
        self.divergence_epoch_loss = 0
        self.val_divergence_epoch_loss = 0
        self.burned_epoch_loss = 0
        self.val_burned_epoch_loss = 0
        self.last_mu = None
        self.last_logvar = None
        
    def encode(self, x):
        # Encoder
        x = self.encoder(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)   
        return x_mu, x_logvar

    def decode(self, x):
        # The full embedding is used for the reconstruction
        # Decoder
        x = self.decoder(x)
        return torch.sigmoid(x) if self.is_sigmoid else x
    
    def predict_burned(self, x):
        # Half of the embedding is used for the burned % prediction
        x = x[:, self.rec_dim:]
        r = torch.sigmoid(self.burned_predictor(x))
        return r
    
    def forward(self, x, r):
        latent_mu, latent_logvar = self.encode(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decode(latent)
        r = self.predict_burned(latent)
        self.last_mu = latent_mu
        self.last_logvar = latent_logvar
        return x_recon, r
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.normal(torch.zeros(std.shape), self.distribution_std).to(self.device)
            sample = mu + (eps * std)
            return sample
        else:
            return mu
    
    def vae_loss(self, recon_x, x, mu, logvar, pred_r, r):
        # recon_x is the probability of a multivariate Bernoulli distribution p.
        # -log(p(x)) is then the pixel-wise binary cross-entropy.
        # Averaging or not averaging the binary cross-entropy over all pixels here
        # is a subtle detail with big effect on training, since it changes the weight
        # we need to pick for the other loss term by several orders of magnitude.
        # Not averaging is the direct implementation of the negative log likelihood,
        # but averaging makes the weight of the other loss term independent of the image resolution.
        if self.is_sigmoid:
            if not self.not_reduced:
                recon_loss = F.binary_cross_entropy(recon_x.view(-1, 400), x[:, 0, :, :].view(-1, 400), reduction='sum')
            else:
                recon_loss = self.criterion_vae(recon_x, x[:, 0, :, :])
        else:
            recon_loss = self.criterion_vae(recon_x, x[:, 0, :, :])
        # KL-divergence between the prior distribution over latent vectors
        # (the one we are going to sample from when generating new images)
        # and the distribution estimated by the generator for the given image.
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        reward_loss = self.criterion_r(pred_r, r) * self.alpha
        return recon_loss, kldivergence, reward_loss, recon_loss + self.variational_beta * kldivergence + reward_loss

    def loss(self, output, x, r):
        x_rec, r_pred = output
        mu, logvar = (self.last_mu, self.last_logvar)
        loss_1, loss_2, loss_3, loss = self.vae_loss(x_rec, x, mu, logvar, r_pred, r)
        self.reconstruction_epoch_loss += loss_1.item()
        self.divergence_epoch_loss += loss_2.item()
        self.burned_epoch_loss += loss_3.item()
        self.epoch_loss += loss.item()
        return loss

    
    def val_loss(self, output, x, r):
        x_rec, r_pred = output
        mu, logvar = (self.last_mu, self.last_logvar)
        loss_1, loss_2, loss_3, loss = self.vae_loss(x_rec, x, mu, logvar, r_pred, r)
        self.val_reconstruction_epoch_loss += loss_1.item()
        self.val_divergence_epoch_loss += loss_2.item()
        self.val_burned_epoch_loss += loss_3.item()
        self.val_epoch_loss += loss.item()
        return loss
        
    def step(self):
        self.optimizer_1.step()
        self.optimizer_2.step()
    
    def zero_grad(self):
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()

    def reset_losses(self):
        self.training_loss.append(self.epoch_loss/self.n)
        self.validation_loss.append(self.val_epoch_loss/self.m)
        self.reconstruction_training_loss.append(self.reconstruction_epoch_loss/self.n)
        self.reconstruction_validation_loss.append(self.val_reconstruction_epoch_loss/self.m)
        self.divergence_training_loss.append(self.divergence_epoch_loss/self.n)
        self.divergence_validation_loss.append(self.val_divergence_epoch_loss/self.m)
        self.burned_training_loss.append(self.burned_epoch_loss/self.n)
        self.burned_validation_loss.append(self.val_burned_epoch_loss/self.m)
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = 0
        self.val_reconstruction_epoch_loss = 0
        self.divergence_epoch_loss = 0
        self.val_divergence_epoch_loss = 0
        self.burned_epoch_loss = 0
        self.val_burned_epoch_loss = 0
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
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_lr1={self.lr1}_lr2={self.lr2}_not_reduced={self.not_reduced}_variational_beta={self.variational_beta}_distribution_std={self.distribution_std}_alpha={self.alpha}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.reconstruction_training_loss[1:], label='training loss')
        plt.plot(self.reconstruction_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/reconstruction_loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_lr1={self.lr1}_lr2={self.lr2}_not_reduced={self.not_reduced}_variational_beta={self.variational_beta}_distribution_std={self.distribution_std}_alpha={self.alpha}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.divergence_training_loss[1:], label='training loss')
        plt.plot(self.divergence_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/divergence_loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_lr1={self.lr1}_lr2={self.lr2}_not_reduced={self.not_reduced}_variational_beta={self.variational_beta}_distribution_std={self.distribution_std}_alpha={self.alpha}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.burned_training_loss[1:], label='training loss')
        plt.plot(self.burned_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/burned_loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_lr1={self.lr1}_lr2={self.lr2}_not_reduced={self.not_reduced}_variational_beta={self.variational_beta}_distribution_std={self.distribution_std}_alpha={self.alpha}.png")
        
    def calc_test_loss(self, output, images, r):
        return self.loss(output, images, r)