import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
class FireAutoencoder_reward(nn.Module):
    def __init__(self, params):
        super(FireAutoencoder_reward, self).__init__()
        self.c = params["capacity"]
        self.name = "AE_Reward"
        self.instance = params["instance"]
        self.latent_dims = params["latent_dims"]
        kernel_size = 4
        stride = 2
        padding = 1
        input_size = params["input_size"]
        self.dim_1 = int((input_size - kernel_size + 2*padding)/2 + 1)
        self.dim_2 = int((self.dim_1 - kernel_size + 2*padding)/2 + 1)
        self.is_sigmoid = params["sigmoid"]
        self.scale = params["scale"]
        self.lr1 = params["lr1"]
        self.lr2 = params["lr2"]
        self.lr3 = params["lr3"]
        self.normalize = params["normalize"]
        self.weight_decay = params["weight_decay"]
        self.temperature_1 = params["temperature_1"]
        self.temperature_2 = params["temperature_2"]
        # Grouping parameters for different optimizers
        self.encoder_params = []
        self.decoder_params = []
        self.regression_params = []

        # Encoder layers:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding) # (64, 10, 10)
        self.bn_1 = nn.BatchNorm2d(self.c)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=kernel_size, stride=stride, padding=padding) # (128, 5, 5)
        self.bn_2 = nn.BatchNorm2d(self.c*2)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=self.latent_dims*(self.dim_2**2), out_features = self.latent_dims)
        self.bn_3 = nn.BatchNorm1d(self.latent_dims)
        self.encoder_params.extend(self.conv1.parameters())
        self.encoder_params.extend(self.conv2.parameters())
        self.encoder_params.extend(self.fc.parameters())
        # Decoder layers:
        self.fc_2 = nn.Linear(in_features=self.latent_dims, out_features=self.latent_dims*(self.dim_2**2))
        self.bn_1_2 = nn.BatchNorm1d(self.latent_dims*(self.dim_2**2))
        self.conv1_2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_2_2 = nn.BatchNorm2d(self.c)
        self.dp2 = nn.Dropout(p=0.2)
        self.conv2_2 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.decoder_params.extend(self.conv1_2.parameters())
        self.decoder_params.extend(self.conv2_2.parameters())
        self.decoder_params.extend(self.fc_2.parameters())
        if self.is_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self.scale = nn.Sigmoid()

        # Reward predictor:
        self.fc_r1 = nn.Linear(in_features=self.latent_dims, out_features=128)
        self.bn_r1 = nn.BatchNorm1d(128)
        self.fc_r2 = nn.Linear(in_features=128, out_features=64)
        self.bn_r2 = nn.BatchNorm1d(64)
        self.dpr1 = nn.Dropout(p=0.2)
        self.fc_r3 = nn.Linear(in_features=64, out_features=32)
        self.bn_r3 = nn.BatchNorm1d(32)
        self.fc_r4 = nn.Linear(in_features=32, out_features=1)
        self.regression_params.extend(self.fc_r1.parameters())
        self.regression_params.extend(self.fc_r2.parameters())
        self.regression_params.extend(self.fc_r3.parameters())
        self.regression_params.extend(self.fc_r4.parameters())

        # Inicialización de parámetros:
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')

        # We create optimizers for each task:
        self.optimizer1 = torch.optim.Adam(self.encoder_params, lr = self.lr1, weight_decay=self.weight_decay)
         # We create optimizers for each task
        self.optimizer2 = torch.optim.Adam(self.decoder_params, lr = self.lr2, weight_decay=self.weight_decay)
         # We create optimizers for each task
        self.optimizer3 = torch.optim.Adam(self.regression_params, lr = self.lr3, weight_decay=self.weight_decay)

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
        self.T1 = self.temperature_1
        self.T2 = self.temperature_2

        if self.is_sigmoid:
            self.sigmoid = nn.Sigmoid()
            self.criterion_1 = nn.BCELoss()
        else:
            self.criterion_1 = nn.MSELoss()
        self.criterion_2 = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode(self, x):
        u1 = self.bn_1(F.relu(self.conv1(x)))
        u2 = self.dp1(self.bn_2(F.relu(self.conv2(u1))))
        flat = u2.view(x.size(0), -1)
        output = self.bn_3(F.relu(self.fc(flat)))
        return output

    
    def decode(self, x):
        h = self.bn_1_2(F.relu(self.fc_2(x)))
        mat = h.view(x.size(0), self.c*2, self.dim_2, self.dim_2)
        u1 = self.dp2(self.bn_2_2(F.relu(self.conv1_2(mat))))
        u2 = self.sigmoid(self.conv2_2(u1)) if self.is_sigmoid else F.relu(self.conv2_2(u1))
        self.last_layer1 = u2
        return u2
    
    def predict_reward(self, x):
        h1 = self.bn_r1(F.relu(self.fc_r1(x)))
        h2 = self.dpr1(self.bn_r2(F.relu(self.fc_r2(h1))))
        h3 = self.bn_r3(F.relu(self.fc_r3(h2)))
        reward = self.fc_r4(h3)
        self.last_layer2 = reward
        return self.scale(reward) if self.normalize else reward

    def forward(self, x, r):
        embedding = self.encode(x)
        return self.decode(embedding), self.predict_reward(embedding)
    
    def compute_loss(self, output, x, r):
        output_x = output[0]
        output_r = output[1]
        loss_1 = self.criterion_1(output_x, x)
        loss_2 = self.criterion_2(output_r.squeeze(), r)
        loss = torch.exp(loss_1/self.T1) + torch.exp(loss_2/self.T2)
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
    
    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer3.step()

    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
    
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
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_T1={self.T1}_T2={self.T2}_lr1={self.lr1}_lr2={self.lr2}_lr3={self.lr3}_normalize={self.normalize}_weight_decay={self.weight_decay}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.reconstruction_training_loss[1:], label='training loss')
        plt.plot(self.reconstruction_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/reconstruction_loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_T1={self.T1}_T2={self.T2}_lr1={self.lr1}_lr2={self.lr2}_lr3={self.lr3}_normalize={self.normalize}_weight_decay={self.weight_decay}.png")

        plt.ion()
        fig = plt.figure()
        plt.plot(self.regression_training_loss[1:], label='training loss')
        plt.plot(self.regression_validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"experiments/{self.instance}/train_stats/{self.name}/regression_loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}_sigmoid={self.is_sigmoid}_T1={self.T1}_T2={self.T2}_lr1={self.lr1}_lr2={self.lr2}_lr3={self.lr3}_normalize={self.normalize}_weight_decay={self.weight_decay}.png")

    def calc_test_loss(self, output, images, r):
        return self.loss(output, images, r)