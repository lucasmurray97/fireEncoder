import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
class FireAutoencoder_reward(nn.Module):
    def __init__(self, capacity, input_size, latent_dims, sigmoid=False, scale = 10e-5, temperature = 10):
        super(FireAutoencoder_reward, self).__init__()
        self.c = capacity
        self.name = "AE_Reward"
        self.latent_dims = latent_dims
        kernel_size = 4
        stride = 2
        padding = 1
        self.dim_1 = int((input_size - kernel_size + 2*padding)/2 + 1)
        self.dim_2 = int((self.dim_1 - kernel_size + 2*padding)/2 + 1)
        self.is_sigmoid = sigmoid
        self.scale = scale
        self.last_layer1 = None
        self.last_layer2 = None
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
        self.scale = nn.Sigmoid()

        # Reward predictor:
        self.fc_r1 = nn.Linear(in_features=latent_dims, out_features=128)
        self.fc_r2 = nn.Linear(in_features=128, out_features=64)
        self.fc_r3 = nn.Linear(in_features=64, out_features=32)
        self.fc_r4 = nn.Linear(in_features=32, out_features=1)
        
        # Inicialización de parámetros:
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')

        # Loss weigthts:
        self.sigma_1 = nn.parameter.Parameter(torch.Tensor([1]))
        self.sigma_2 = nn.parameter.Parameter(torch.Tensor([1]))
        self.epsilon = 1e-8
        # Losses holders:
        self.training_loss = []
        self.validation_loss = []
        self.reconstruction_training_loss = []
        self.reconstruction_validation_loss = []
        self.regression_training_loss = []
        self.regression_validation_loss = []
        self.m = 0
        self.n = 0
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = 0
        self.val_reconstruction_epoch_loss = 0
        self.regression_epoch_loss = 0
        self.val_regression_epoch_loss = 0
        self.T = temperature

        if self.is_sigmoid:
            self.sigmoid = nn.Sigmoid()
            self.criterion_1 = nn.BCELoss()
        else:
            self.criterion_1 = nn.MSELoss()
        self.criterion_2 = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.last_layer1 = u2
        return u2
    
    def predict_reward(self, x):
        h1 = F.relu(self.fc_r1(x))
        h2 = F.relu(self.fc_r2(h1))
        h3 = F.relu(self.fc_r3(h2))
        reward = self.fc_r4(h3)
        self.last_layer2 = reward
        return self.scale(reward)

    def forward(self, x, r):
        embedding = self.encode(x)
        return self.decode(embedding), self.predict_reward(embedding)
    
    def compute_loss(self, output, x, r):
        output_x = output[0]
        output_r = output[1]
        loss_1 = self.criterion_1(output_x, x)
        loss_2 = self.criterion_2(output_r.squeeze(), r)
        loss = torch.exp(loss_1/self.T) + torch.exp(loss_2/self.T)
        return loss_1, loss_2, loss
    
    def loss(self, output, x, r):
        loss_1, loss_2, loss = self.compute_loss(output, x, r)
        self.reconstruction_epoch_loss += loss_1.item()
        self.regression_epoch_loss += loss_2.item()
        self.epoch_loss += loss.item()
        return loss

    
    def val_loss(self, output, x, r):
        loss_1, loss_2, loss = self.compute_loss(output, x, r)
        self.val_reconstruction_epoch_loss += loss_1.item()
        self.val_regression_epoch_loss += loss_2.item()
        self.val_epoch_loss += loss.item()
        return loss
    
    def show_grads(self):
        grads_1 = 0
        for i in self.fc_r4.parameters():
            grads_1 += i.grad.sum()
        grads_2 = 0
        for i in self.conv2_2.parameters():
            grads_2 += i.grad.sum()

    def reset_losses(self):
        self.training_loss.append(self.epoch_loss/self.n)
        self.validation_loss.append(self.val_epoch_loss/self.m)
        self.reconstruction_training_loss.append(self.reconstruction_epoch_loss/self.n)
        self.reconstruction_validation_loss.append(self.val_reconstruction_epoch_loss/self.m)
        self.regression_training_loss.append(self.regression_epoch_loss/self.n)
        self.regression_validation_loss.append(self.val_regression_epoch_loss/self.m)
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = 0
        self.val_reconstruction_epoch_loss = 0
        self.regression_epoch_loss = 0
        self.val_regression_epoch_loss = 0
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
        plt.savefig(f"experiments/train_stats/{self.name}/loss_homo_2_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_{self.T}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.reconstruction_training_loss[1:], label='training loss')
        plt.plot(self.reconstruction_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/train_stats/{self.name}/reconstruction_loss_homo_2_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_{self.T}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.regression_training_loss[1:], label='training loss')
        plt.plot(self.regression_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/train_stats/{self.name}/regression_loss_homo_2_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_{self.T}.png")

    def calc_test_loss(self, output, images, r):
        return self.loss(output, images, r)