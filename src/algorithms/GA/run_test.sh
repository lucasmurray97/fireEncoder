#!/usr/bin/env bash
set -euo pipefail

REPEATS=6
ITERS=20

# LRS=("1e-7" "1e-8")              # fine-tune LR for CCVAE
# EPOCHS=(2)
# STEPS=(2 3)                    # inner mutation steps
# TRUST_R=("" 3 5)                 # trust region radius
# ETA_MUT=(0.5 0.9 1.2)           # mutation LR
ALPHA=(0.5 0.1 0.8)
MUT_RATE=(0.5 0.2 0.8) 
V=("v1" "v2")

# for lr in "${LRS[@]}"; do
  # for ep in "${EPOCHS[@]}"; do
    for alpha in "${ALPHA[@]}"; do
      for mut in "${MUT_RATE[@]}"; do
        for ver in "${V[@]}"; do
          # for mx in "${MAX_NORM[@]}"; do
            python train.py \
              --n_repeats "$REPEATS" \
              --iters "$ITERS" \
              --algorithm "$ver" \
              --variational_beta 0.1 \
              --distribution_std 1.0 \
              --initial_population 0.002 \
              --population_size 300 \
              --alpha "$alpha" \
              --mutation_rate "$mut" \
              --net ccvae \
              # --finetune \
              # --lr "$lr" \
              # --epochs "$ep" \
              # --steps "$st" \
              # --trust_region "$r" \
              # --score_lr "$eta" \
              # --max_norm "$mx"
          done
        done
      done
#     done
#   done
# done