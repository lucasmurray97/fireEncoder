from abstract_ga import Abstract_Genetic_Algorithm
from variational_GA import Variational_GA_V1
from variational_GA_v2 import Variational_GA_V2
from vainilla_ga import Vainilla_GA
import torch
import sys
sys.path.append("../../")
from networks.vae import VAE

latent_dims = 256
capacity = latent_dims//2
input_size = 20
epochs = 100
sigmoid = True
instance = "homo_2"
lr1 = 1e-5
not_reduced = False
variational_beta = 0.01
distribution_std = 1.
params = {
    "input_size": input_size,
    "latent_dims": latent_dims,
    "capacity": capacity,
    "epochs": epochs,
    "sigmoid": sigmoid,
    "instance": instance,
    "lr1": lr1,
    "not_reduced": not_reduced,
    "variational_beta": variational_beta,
    "distribution_std": distribution_std,
}
net = VAE(params)
print(net.training)
net.load_state_dict(torch.load(f'../../weights/homo_2/VAE/sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}_T1=100_T2=100_lr1={lr1}_lr2=0.0001_lr3=0.0001_normalize=False_weight_decay=0_not_reduced={not_reduced}_variational_beta={variational_beta}_distribution_std={distribution_std}.pth', map_location=torch.device('cpu') ))
a = Vainilla_GA(net)
a.initialize_population(alpha=0.01)
a.train(n_iter=100)

a = Variational_GA_V1(net)
a.initialize_population(alpha=0.01)
a.train(n_iter=100)

a = Variational_GA_V2(net)
a.initialize_population(alpha=0.01)
a.train(n_iter=100)

