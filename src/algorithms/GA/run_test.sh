#!/bin/bash
python train.py --n_repeats 6 --iters 20 --algorithm cd --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae --cond_thresh 0.76
#python train.py --n_repeats 6 --iters 20 --algorithm gd --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae --gradient_step 2

#python train.py --n_repeats 6 --iters 20 --algorithm cd --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae --cond_thresh 0.75
python train.py --n_repeats 6 --iters 20 --algorithm gd --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae --gradient_step 1
python train.py --n_repeats 6 --iters 20 --algorithm cd --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae --cond_thresh 0.77
#python train.py --n_repeats 1 --iters 3 --algorithm v1 --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
#python train.py --n_repeats 1 --iters 3 --algorithm v2 --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
#python train.py --n_repeats 6 --iters 20 --algorithm md --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
#python train.py --n_repeats 6 --iters 20 --algorithm gd --variational_beta 0.1 --distribution_std 0.2 --initial_population 0.002 --population_size 300 --alpha 0.8 --mutation_rate 0.5  --net ccvae
