import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
class FireAutoencoder(nn.Module):
    def __init__(self, capacity, input_size, latent_dims, sigmoid=False, instance = "homo_2"):
        super(FireAutoencoder, self).__init__()
        self.name = "AE"
        self.instance= instance
        self.latent_dims = latent_dims
        self.c = capacity
        kernel_size = 4
        stride = 2
        padding = 1
        self.dim_1 = int((input_size - kernel_size + 2*padding)/2 + 1)
        self.dim_2 = int((self.dim_1 - kernel_size + 2*padding)/2 + 1)
        self.is_sigmoid = sigmoid
        # Encoder layers:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding) # (64, 10, 10)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=kernel_size, stride=stride, padding=padding) # (128, 5, 5)
        self.fc = nn.Linear(in_features=latent_dims*(self.dim_2**2), out_features = latent_dims)

        # Decoder layers:
        self.fc_2 = nn.Linear(in_features=latent_dims, out_features=latent_dims*(self.dim_2**2))
        self.conv1_2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_2 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.is_sigmoid:
            self.sigmoid = nn.Sigmoid()
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Inicialización de parámetros:
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')

        # Losses holders:
        self.training_loss = []
        self.validation_loss = []
        self.m = 0
        self.n = 0
        self.epoch_loss = 0
        self.val_epoch_loss = 0

    def encode(self, x):
        u1 = F.relu(self.conv1(x))
        u2 = F.relu(self.conv2(u1))
        flat = u2.view(x.size(0), -1)
        output = self.fc(flat)
        return output

    
    def decode(self, x):
        h = self.fc_2(x)
        mat = h.view(x.size(0), self.c*2, self.dim_2, self.dim_2)
        u1 = F.relu(self.conv1_2(mat))
        u2 = self.sigmoid(self.conv2_2(u1)) if self.is_sigmoid else F.relu(self.conv2_2(u1))
        return u2
        
    def forward(self, x, r):
        return self.decode(self.encode(x))
    
    def loss(self, output, x, r):
        loss = self.criterion(output, x)
        self.epoch_loss += loss.item()
        return loss
    
    def val_loss(self, output, x, r):
        loss = self.criterion(output, x)
        self.val_epoch_loss += loss.item()
        return loss
    
    def reset_losses(self):
        self.training_loss.append(self.epoch_loss/self.n)
        self.validation_loss.append(self.val_epoch_loss/self.m)
        self.epoch_loss = 0
        self.val_epoch_loss = 0
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
        plt.savefig(f"experiments/train_stats/{self.instance}/{self.name}/loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}.png")

    def calc_test_loss(self, output, images, r):
        return self.loss(output, images, r)