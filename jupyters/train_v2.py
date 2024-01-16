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
from networks.autoencoder_reward import FireAutoencoder_reward
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
net = FireAutoencoder_reward(capacity, input_size, latent_dims, sigmoid=sigmoid)
if sigmoid:
    criterion_1 = nn.BCELoss()
else:
    criterion_1 = nn.MSELoss()
criterion_2 = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
# Data loader is built
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Training Loop
training_loss = []
validation_loss = []
reconstruction_training_loss = []
reconstruction_validation_loss = []
regression_training_loss = []
regression_validation_loss = []
net.to(device)
for epoch in tqdm(range(epochs)):
    n = 0
    m = 0
    epoch_loss = 0
    val_epoch_loss = 0
    reconstruction_epoch_loss = 0
    val_reconstruction_epoch_loss = 0
    regression_epoch_loss = 0
    val_regression_epoch_loss = 0
    for x, r_x in train_loader:
        x = x.to(device)
        output, r = net(x)
        reconstruction_loss = criterion_1(output, x)
        regression_loss = criterion_2(r_x.unsqueeze(1), r)
        total_loss = reconstruction_loss + regression_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        reconstruction_epoch_loss += reconstruction_loss.item()
        regression_epoch_loss += regression_loss.item()
        n += 1

    for y, r_y in validation_loader:
        y = y.to(device)
        output, r = net(y)
        val_reconstruction_loss = criterion_1(output, y)
        val_regression_loss = criterion_2(r_y.unsqueeze(1), r)
        val_total_loss = val_reconstruction_loss + val_regression_loss
        val_epoch_loss += val_total_loss.item()
        val_reconstruction_epoch_loss += val_reconstruction_loss.item()
        val_regression_epoch_loss += val_regression_loss.item()
        m+=1
    reconstruction_training_loss.append(reconstruction_epoch_loss/n)
    reconstruction_validation_loss.append(val_reconstruction_epoch_loss/m)
    regression_training_loss.append(regression_epoch_loss/n)
    regression_validation_loss.append(val_regression_epoch_loss/m)
    training_loss.append(epoch_loss/n)
    validation_loss.append(val_epoch_loss/m)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss/n))
    print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, epochs, val_epoch_loss/m))

plt.ion()
fig = plt.figure()
plt.plot(training_loss[1:], label='training loss')
plt.plot(validation_loss[1:], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"train_stats/v2/total_loss_homo_2_sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}.png")

plt.ion()
fig = plt.figure()
plt.plot(reconstruction_training_loss, label='training loss')
plt.plot(reconstruction_validation_loss, label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Loss')
plt.legend()
plt.savefig(f"train_stats/v2/reconstruction_loss_homo_2_sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}.png")

plt.ion()
fig = plt.figure()
plt.plot(regression_training_loss[1:], label='training loss')
plt.plot(regression_validation_loss[1:], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Regression Loss')
plt.legend()
plt.savefig(f"train_stats/v2/regression_loss_homo_2_sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}.png")

path_ = f"./weights/v2/homo_2_sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}.pth"
torch.save(net.state_dict(), path_)

full_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
all_images, all_r = next(iter(full_loader))

images, r = next(iter(full_loader))
output, r_ = net(images.to(device))
loss_1 = criterion_1(output,images)
loss_2 = criterion_2(r_.squeeze(), r)
total_loss = loss_1 + loss_2
f = open("train_stats/v2/test_losses.txt", "a")
f.write(str(latent_dims)+','+str(capacity)+','+str(total_loss.item())+"\n")
