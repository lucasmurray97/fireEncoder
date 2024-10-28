from abstract_ga import Abstract_Genetic_Algorithm
from variational_GA import Variational_GA_V1
from variational_GA_v2 import Variational_GA_V2
from vainilla_ga import Vainilla_GA
import torch
import sys
import argparse
sys.path.append("../../")
from networks.vae import VAE
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--n_repeats', type=int, required=True)
parser.add_argument('--initial_population', type=float, required=True)
parser.add_argument('--population_size', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--mutation_rate', type=float, required=True)
parser.add_argument('--iters', type=int, required=True, default = 100)
parser.add_argument('--variational_beta', type=float, default=1)
parser.add_argument('--distribution_std', type=float, default=1)

args = parser.parse_args()
# Params
params = {}
algorithm = args.algorithm
n_repeats = args.n_repeats
initial_population = args.initial_population
population_size = args.population_size
alpha = args.alpha
mutation_rate = args.mutation_rate
variational_beta = args.variational_beta
distribution_std = args.distribution_std
iters = args.iters
latent_dims = 256
capacity = latent_dims//2
input_size = 20
epochs = 100
sigmoid = True
instance = "homo_2"
lr1 = 1e-5
not_reduced = False
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
net.load_state_dict(torch.load(f'../../weights/homo_2/VAE/sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}_T1=100_T2=100_lr1={lr1}_lr2=0.0001_lr3=0.0001_normalize=False_weight_decay=0_not_reduced={not_reduced}_variational_beta={variational_beta}_distribution_std={distribution_std}.pth', map_location=torch.device('cpu') ))
method = None
if algorithm == "v1":
    method = Variational_GA_V1(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)
elif algorithm == "v2":
    method = Variational_GA_V2(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)
else:
    method = Vainilla_GA(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)

method.train(n_iter=iters, n_repeats=n_repeats)



