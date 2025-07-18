import torch
import sys
import argparse
from algorithms import (
    Variational_GA_V1,
    Variational_GA_V2,
    Variational_GA_V1_CCVAE,
    Variational_GA_V2_CCVAE,
    Variational_GA_MD_CCVAE,
    Variational_GA_GD_CCVAE,
    Variational_GA_GD_V2_CCVAE,
    Variational_GA_CD_CCVAE,
    Vainilla_GA,
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
parser.add_argument('--gradient_step', type=int, default=2)
parser.add_argument('--cond_thresh', type=float, default=0.75)


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
gradient_step = args.gradient_step
cond_thresh = args.cond_thresh
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
    "gradient_step": gradient_step,
    "cond_thresh": cond_thresh
}

if network == "vae" and algorithm == "v1":
    net = VAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/VAE/sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}_T1=100_T2=100_lr1={lr1}_lr2=0.0001_lr3=0.0001_normalize=False_weight_decay=0_not_reduced={not_reduced}_variational_beta={variational_beta}_distribution_std={distribution_std}.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_V1(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)
elif network == "vae" and algorithm == "v2":
    net = VAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/VAE/sub20x20_latent={latent_dims}_capacity={capacity}_{epochs}_sigmoid={sigmoid}_T1=100_T2=100_lr1={lr1}_lr2=0.0001_lr3=0.0001_normalize=False_weight_decay=0_not_reduced={not_reduced}_variational_beta={variational_beta}_distribution_std={distribution_std}.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_V2(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)
elif network == "ccvae" and algorithm == "v1":
    net = CCVAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/CCVAE/sub20x20_latent=256_capacity=128_100_sigmoid=True_T1=100_T2=100_lr1=1e-05_lr2=1e-05_lr3=0.0001_normalize=False_weight_decay=0_not_reduced=False_variational_beta=0.1_distribution_std=0.2_alpha=100000.0.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_V1_CCVAE(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)
elif network == "ccvae" and algorithm == "v2":
    net = CCVAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/CCVAE/sub20x20_latent=256_capacity=128_100_sigmoid=True_T1=100_T2=100_lr1=1e-05_lr2=1e-05_lr3=0.0001_normalize=False_weight_decay=0_not_reduced=False_variational_beta=0.1_distribution_std=0.2_alpha=100000.0.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_V2_CCVAE(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)
elif network == "ccvae" and algorithm == "md":
    net = CCVAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/CCVAE/sub20x20_latent=256_capacity=128_100_sigmoid=True_T1=100_T2=100_lr1=1e-05_lr2=1e-05_lr3=0.0001_normalize=False_weight_decay=0_not_reduced=False_variational_beta=0.1_distribution_std=0.2_alpha=100000.0.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_MD_CCVAE(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)
elif network == "ccvae" and algorithm == "gd":
    net = CCVAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/CCVAE/sub20x20_latent=256_capacity=128_100_sigmoid=True_T1=100_T2=100_lr1=1e-05_lr2=1e-05_lr3=0.0001_normalize=False_weight_decay=0_not_reduced=False_variational_beta=0.1_distribution_std=0.2_alpha=100000.0.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_GD_CCVAE(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population, gradient_step=gradient_step)
elif network == "ccvae" and algorithm == "gd_v2":
    net = CCVAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/CCVAE/sub20x20_latent=256_capacity=128_100_sigmoid=True_T1=100_T2=100_lr1=1e-05_lr2=1e-05_lr3=0.0001_normalize=False_weight_decay=0_not_reduced=False_variational_beta=0.1_distribution_std=0.2_alpha=100000.0.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_GD_V2_CCVAE(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population, gradient_step=gradient_step)
elif network == "ccvae" and algorithm == "cd":
    net = CCVAE(params)
    net.load_state_dict(torch.load(f'../../weights/homo_2/CCVAE/sub20x20_latent=256_capacity=128_100_sigmoid=True_T1=100_T2=100_lr1=1e-05_lr2=1e-05_lr3=0.0001_normalize=False_weight_decay=0_not_reduced=False_variational_beta=0.1_distribution_std=0.2_alpha=100000.0.pth', map_location=torch.device('cpu') ))
    method = Variational_GA_CD_CCVAE(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population, cond_thresh=cond_thresh)
else:
    net = None
    method = Vainilla_GA(net, alpha=alpha, mutation_rate=mutation_rate, population_size=population_size, initial_population=initial_population)

method.train(n_iter=iters, n_repeats=n_repeats)



