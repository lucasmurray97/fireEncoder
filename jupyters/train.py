import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import pickle
from matplotlib import pyplot as plt
from utils.utils import MyDataset, show_image, visualise_output
import sys
sys.path.append("..")
from networks.autoencoder import FireAutoencoder
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
# Params
latent_dims = args.latent_dim
capacity = latent_dims//2
use_gpu =  True
input_size = 20
epochs = args.epochs
sigmoid = args.sigmoid
# Dataset is loaded
dataset = MyDataset(root='../data/complete_random/homo_2/Sub20x20_full_grid_.pkl',
                             tform=lambda x: torch.from_numpy(x, dtype=torch.float))

train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.9, 0.05, 0.05])

# Network is constructed
net = FireAutoencoder(capacity, input_size, latent_dims, sigmoid=sigmoid)
if sigmoid:
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
# Data loader is built
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Training Loop
training_loss = []
validation_loss = []
net.to(device)
for epoch in tqdm(range(epochs)):
    n = 0
    m = 0
    epoch_loss = 0
    val_epoch_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        output = net(x)
        loss = criterion(output,x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n+=1

    for y, _ in validation_loader:
        y = y.to(device)
        output = net(y)
        val_loss = criterion(output,y)
        val_epoch_loss += val_loss.item()
        m+=1
    training_loss.append(epoch_loss/n)
    validation_loss.append(val_epoch_loss/m)

plt.ion()
fig = plt.figure()
plt.plot(training_loss, label='training loss')
plt.plot(validation_loss, label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"train_stats/loss_homo_2_sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}.png")

path_ = f"./weights/homo_2_sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}.pth"
torch.save(net.state_dict(), path_)

full_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
all_images, all_r = next(iter(full_loader))

images, r = next(iter(full_loader))
output = net(images.to(device))
loss = criterion(output,images)
f = open("train_stats/test_losses.txt", "a")
f.write(str(latent_dims)+','+str(capacity)+','+str(loss.item())+"\n")
