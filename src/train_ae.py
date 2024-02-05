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
from networks.autoencoder_reward import FireAutoencoder_reward
from networks.utils import EarlyStopper
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--network', type=str, default="AE")
parser.add_argument('--lr1', type=float, default=0.0001)
parser.add_argument('--lr2', type=float, default=0.0001)
parser.add_argument('--lr3', type=float, default=0.0001)
parser.add_argument('--temperature_1', type=float, default = 100)
parser.add_argument('--temperature_2', type=float, default = 100)
parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--weight_decay', type=float, default=0)

args = parser.parse_args()
# Params
latent_dims = args.latent_dim
capacity = latent_dims//2
use_gpu =  True
input_size = 20
epochs = args.epochs
sigmoid = args.sigmoid
network = args.network
lr1 = args.lr1
lr2 = args.lr2
lr3 = args.lr3
temperature_1 = args.temperature_1
temperature_2 = args.temperature_2
normalize = args.normalize
weight_decay = args.weight_decay
# Dataset is loaded
dataset = MyDataset(root='../data/complete_random/homo_2/Sub20x20_full_grid_.pkl',
                             tform=lambda x: torch.from_numpy(x, dtype=torch.float), normalize=normalize)

train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.9, 0.05, 0.05])

# Network is constructed
nets = {
    "AE": FireAutoencoder,
    "AE_Reward": FireAutoencoder_reward,
}
net = nets[network](capacity, input_size, latent_dims, sigmoid=sigmoid, temperature_1=temperature_1, temperature_2=temperature_2, lr1 = lr1, lr2 = lr2, lr3 = lr3, normalize = normalize, weight_decay=weight_decay)
# Data loader is built
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
# Training Loop
net.to(net.device)
early_stopper = EarlyStopper(patience=5, min_delta=0.01)
for epoch in tqdm(range(epochs)):
    for x, r_x in train_loader:
        # print(r_x)
        net.zero_grad()
        x = x.to(net.device)
        r_x = r_x.to(net.device)
        output = net(x, r_x)
        loss = net.loss(output, x, r_x)
        loss.backward()   
        # net.show_grads()
        net.step()
        net.n+=1

    for y, r_y in validation_loader:
        y = y.to(net.device)
        r_y = r_y.to(net.device)
        output = net(y, r_y)
        val_loss = net.val_loss(output,y, r_y)
        net.m+=1
    if early_stopper.early_stop(net.val_epoch_loss):             
      print("Early stoppage at epoch:", epoch)
      break
    net.reset_losses()
    
net.plot_loss(epochs)

path_ = f"./weights/{network}/homo_2_sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}.pth"
torch.save(net.state_dict(), path_)
net.eval()
full_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
images, r = next(iter(full_loader))
output = net(images.to("cpu"), r.to("cpu"))
loss = net.calc_test_loss(output, images, r)
f = open(f"experiments/train_stats/{network}/test_losses.txt", "a")
f.write(str(latent_dims)+','+str(capacity)+','+str(loss.item())+"\n")
