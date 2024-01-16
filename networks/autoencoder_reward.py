import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np

class FireAutoencoder_reward(nn.Module):
    def __init__(self, capacity, input_size, latent_dims, sigmoid=False):
        super(FireAutoencoder_reward, self).__init__()
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
    
    def predict_reward(self, x):
        h1 = F.relu(self.fc_r1(x))
        h2 = F.relu(self.fc_r2(h1))
        h3 = F.relu(self.fc_r3(h2))
        reward = self.fc_r4(h3)
        return reward

    def forward(self, x):
        embedding = self.encode(x)
        return self.decode(embedding), self.predict_reward(embedding)
