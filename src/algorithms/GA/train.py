import torch
import sys
import argparse
from algorithms import (
    Vainilla_GA,
    Variational_GA,
    Variational_GA_CCVAE
)
sys.path.append("../../")
from networks.vae import VAE
from networks.ccvae import CCVAE


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
parser.add_argument('--net', type=str, default="vae")
parser.add_argument('--steps', type=int, default=2)
parser.add_argument('--cond_thresh', type=float, default=0.75)
parser.add_argument('--finetune', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epochs_ft', type=int, default=1)
parser.add_argument('--score_lr', type=float, default=0.2)
parser.add_argument('--trust_region', type=float, default=None)
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
network = args.net
iters = args.iters
steps = args.steps
cond_thresh = args.cond_thresh
finetune = args.finetune
lr = args.lr
score_lr = args.score_lr
trust_region = args.trust_region
epochs_ft = args.epochs_ft
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
    "lr2": lr1,
    "not_reduced": not_reduced,
    "variational_beta": variational_beta,
    "distribution_std": distribution_std,
    "use_gpu": True,
    "latent_portion": 0.5,
    "alpha": 1e5,
    "steps": steps,
    "cond_thresh": cond_thresh,
}

if network == "vae":
    net = VAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/VAE/sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}_T1=100_T2=100_lr1={lr1}_lr2=0.0001_lr3=0.0001_normalize=False_weight_decay=0_not_reduced={not_reduced}_variational_beta={variational_beta}_distribution_std={distribution_std}.pth', map_location=torch.device('cpu') ))
    method = Variational_GA(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population, finetune=finetune, lr=lr, epochs=epochs_ft, strategy=algorithm)
elif network == "ccvae":
    net = CCVAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/CCVAE/sub20x20_latent=256_capacity=128_100_sigmoid=True_T1=100_T2=100_lr1=1e-05_lr2=1e-05_lr3=0.0001_normalize=False_weight_decay=0_not_reduced=False_variational_beta=0.1_distribution_std=0.2_alpha=100000.0.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_CCVAE(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population, finetune=finetune, lr=lr, epochs=epochs_ft, strategy=algorithm, steps=steps, eta_mut=score_lr, trust_R=trust_region)
else:
    net = None
    method = Vainilla_GA(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)

method.train(n_iter=iters, n_repeats=n_repeats)



