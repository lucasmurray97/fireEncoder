#!/bin/bash
python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.005 --population_size 200 --alpha 0.5 --mutation_rate 0.2 --net ccvae
python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.01 --population_size 200 --alpha 0.5 --mutation_rate 0.5  --net ccvae

python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.01 --population_size 300 --alpha 0.5 --mutation_rate 0.2  --net ccvae
python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.01 --population_size 300 --alpha 0.5 --mutation_rate 0.5  --net ccvae

##
python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.01 --population_size 200 --alpha 0.8 --mutation_rate 0.2  --net ccvae
python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.01 --population_size 200 --alpha 0.8 --mutation_rate 0.5  --net ccvae

python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.01 --population_size 300 --alpha 0.8 --mutation_rate 0.2  --net ccvae
python train.py --n_repeats 8 --iters 40 --algorithm v1 --variational_beta 0.1 --distribution_std 1. --initial_population 0.01 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae