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
from networks.ann_reward import ANN
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy
import argparse
from tqdm import tqdm
dataset = MyDataset(root='../data/complete_random/homo_2/Sub20x20_full_grid_.pkl',
                             tform=lambda x: torch.from_numpy(x, dtype=torch.float))

parser = argparse.ArgumentParser()
parser.add_argument('--capacity', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--lr', type=float, required=True, default = 0.0001)
args = parser.parse_args()
# Params
capacity = args.capacity
use_gpu =  True
input_size = 20
epochs = args.epochs
sigmoid = args.sigmoid
lr = args.lr
latent_dims = 256
train_epochs = 100
net = FireAutoencoder(latent_dims//2, input_size, latent_dims, sigmoid)
net.load_state_dict(torch.load(f'weights/v1/homo_2_sub20x20_latent={latent_dims}_capacity={latent_dims//2}_{train_epochs}_sigmoid={sigmoid}.pth', map_location=torch.device('cpu')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.9, 0.05, 0.05])

batch = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch, shuffle=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=False)

full_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

all_images, all_r = next(iter(full_loader))
net.to(device)
reward_ann = ANN(latent_dims, capacity)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(reward_ann.parameters(), lr = lr)
reward_ann.to(device)
training_loss = []
validation_loss = []
for epoch in tqdm(range(epochs)):
    n = 0
    m = 0
    epoch_loss = 0
    val_epoch_loss = 0
    for x, r in train_loader:
        r = r.to(device)
        x = x.to(device)
        embedding = net.encode(x)
        output = reward_ann(embedding)
        loss = criterion(output.squeeze(), r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n += 1
    
    for y, r in validation_loader:
        r = r.to(device)
        y = y.to(device)
        embedding = net.encode(y)
        output = reward_ann(embedding)
        val_loss = criterion(output.squeeze(),r)
        optimizer.zero_grad()
        val_epoch_loss += val_loss.item()
        m+=1
    training_loss.append(epoch_loss/n)
    validation_loss.append(val_epoch_loss/m)
    # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
    # print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, epochs, val_loss.item()))

plt.ion()
fig = plt.figure()
plt.plot(training_loss[1:], label='training loss')
plt.plot(validation_loss[1:], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f"ann/v1/loss_func_{capacity}_{epochs}_sigmoid={sigmoid}.png")


embeddings = net.encode(all_images.to(device))

with torch.no_grad():
    rewards = reward_ann(embeddings)
plt.ion()
fig = plt.figure()
bins = np.arange(-1020, -500, 10)
plt.hist(rewards.squeeze().detach().numpy(), bins=bins, align='left')
plt.title('Distribución de las recompensas predecidas')
plt.xlabel('Recompensa')
plt.ylabel('Frecuencia')
plt.savefig(f"ann/v1/reward_classes_distr_{capacity}_{epochs}_sigmoid={sigmoid}.png")

def ann(x):
    return reward_ann(torch.from_numpy(x).float()).detach().numpy()

res = scipy.optimize.minimize(ann, x0=np.zeros(latent_dims))
minimum = torch.from_numpy(res.x)
net.float()
solution = net.decode(minimum.float().unsqueeze(0).to(device))
if sigmoid:
    solution[solution>=0.5] = 1
    solution[solution<=0.5] = 0
else:
    solution[solution>0] = 1
with torch.no_grad():
    plt.ion()
    fig = plt.figure()
    plt.title('Reconstrucción del mínimo')
    plt.imshow(solution[0][0].detach().numpy())
    plt.colorbar()
    plt.savefig(f"ann/v1/minimum_decoding_{capacity}_{epochs}_sigmoid={sigmoid}.png")
    

path_ = f"./weights/v1/ann_capacity={capacity}_{epochs}_sigmoid={sigmoid}.pth"
torch.save(reward_ann.state_dict(), path_)

full_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
all_images, all_r = next(iter(full_loader))

images, r = next(iter(full_loader))
output = reward_ann(net.encode(images.to(device)))
loss = criterion(output.squeeze(),r.to(device))
f = open("train_stats/v1/test_losses.txt", "a")
f.write(str(latent_dims)+','+str(capacity)+','+str(loss.item())+"\n")

f = open(f'ann/v1/solution_capacity={capacity}_{epochs}_sigmoid={sigmoid}.txt', 'a')
f.write(str(repr(solution[0][0].detach())))