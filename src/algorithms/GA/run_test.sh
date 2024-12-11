#!/bin/bash
python train.py --n_repeats 6 --iters 20 --algorithm v1 --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.02 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
python train.py --n_repeats 6 --iters 20 --algorithm v2 --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.02 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
python train.py --n_repeats 6 --iters 20 --algorithm md --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.02 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
python train.py --n_repeats 6 --iters 20 --algorithm gd --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.02 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
