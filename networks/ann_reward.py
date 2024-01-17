import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np

class ANN(nn.Module):
    def __init__(self, latent_dims):
        super(ANN, self).__init__()
   
        # Reward predictor:
        self.fc_r1 = nn.Linear(in_features=latent_dims, out_features=128)
        self.fc_r2 = nn.Linear(in_features=128, out_features=64)
        self.fc_r3 = nn.Linear(in_features=64, out_features=32)
        self.fc_r4 = nn.Linear(in_features=32, out_features=1)

    
    def forward(self, x):
        h1 = F.relu(self.fc_r1(x))
        h2 = F.relu(self.fc_r2(h1))
        h3 = F.relu(self.fc_r3(h2))
        reward = self.fc_r4(h3)
        return reward
