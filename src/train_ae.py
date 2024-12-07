import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import pickle
from matplotlib import pyplot as plt
from utils.utils import MyDataset, MyDatasetV2, show_image, visualise_output
import sys
sys.path.append("..")
from networks.autoencoder import FireAutoencoder
from networks.autoencoder_reward import FireAutoencoder_reward
from networks.vae import VAE
from networks.vae_v2 import VAE_V2
from networks.ccvae import CCVAE
from networks.utils import EarlyStopper
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
# For VAE
parser.add_argument('--latent_dim', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--network', type=str, default="AE")
parser.add_argument('--lr1', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--instance', type=str, default="homo_2")
parser.add_argument('--not_reduced', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--variational_beta', type=float, default=1)
parser.add_argument('--distribution_std', type=float, default=1)
parser.add_argument('--data_version', type=int, default=0)

# For CCVAE
parser.add_argument('--latent_portion', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=1000)


# Not for VAE
parser.add_argument('--toy', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--lr2', type=float, default=0.0001)
parser.add_argument('--lr3', type=float, default=0.0001)
parser.add_argument('--temperature_1', type=float, default = 100)
parser.add_argument('--temperature_2', type=float, default = 100)
parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--scale', action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
# Params
params = {}
latent_dims = args.latent_dim
capacity = args.latent_dim//2
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
instance = args.instance
not_reduced= args.not_reduced
variational_beta = args.variational_beta
distribution_std = args.distribution_std
data_version = args.data_version
latent_portion = args.latent_portion
alpha = args.alpha

# Stashing params in a params dictionary
params = {}
params["latent_dims"] = args.latent_dim
params["capacity"] = args.latent_dim//2
params["use_gpu"] =  True
params["input_size"] = 20
params["epochs"] = args.epochs
params["sigmoid"] = args.sigmoid
params["network"] = args.network
params["lr1"] = args.lr1
params["lr2"] = args.lr2
params["lr3"] = args.lr3
params["temperature_1"] = args.temperature_1
params["temperature_2"] = args.temperature_2
params["normalize"] = args.normalize
params["scale"] = args.normalize
params["weight_decay"] = args.weight_decay
params["instance"] = args.instance
params["not_reduced"] = not_reduced
params["variational_beta"] = variational_beta
params["distribution_std"] = distribution_std
params["latent_portion"] = latent_portion
params["alpha"] = alpha

# Dataset is loaded
if data_version == 0:
    dataset = MyDataset(root=f'../data/complete_random/{instance}/')
else:
    dataset = MyDatasetV2(root=f'../data/complete_random/{instance}/')
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.05, 0.05], generator=torch.Generator().manual_seed(42))
# Network is constructed
nets = {
    "AE": FireAutoencoder,
    "AE_Reward": FireAutoencoder_reward,
    "VAE": VAE,
    "VAE_V2": VAE_V2,
    "CCVAE": CCVAE,
}
net = nets[network](params = params)
# Print number of params
print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")
# Data loader is built
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)

# Training Loop
net.to(net.device)
early_stopper = EarlyStopper(patience=5, min_delta=0.01)
for epoch in tqdm(range(epochs)):
    for x, r_x in train_loader:
        net.zero_grad()
        x = x.to(net.device)
        r_x = r_x.to(net.device)
        output = net(x, r_x)
        loss = net.loss(output, x, r_x)
        loss.backward()   
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

path_ = f"./weights/{instance}/{network}/sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}_T1={temperature_1}_T2={temperature_2}_lr1={lr1}_lr2={lr2}_lr3={lr3}_normalize={normalize}_weight_decay={weight_decay}_not_reduced={not_reduced}_variational_beta={variational_beta}_distribution_std={distribution_std}_alpha={alpha}.pth"
torch.save(net.state_dict(), path_)
net.eval()
full_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
images, r = next(iter(full_loader))
output = net(images.to("cpu"), r.to("cpu"))
loss = net.calc_test_loss(output, images, r)
f = open(f"experiments/{instance}/train_stats/{network}/test_losses.txt", "a")
f.write(str(latent_dims)+','+str(capacity)+','+str(loss.item())+"\n")
