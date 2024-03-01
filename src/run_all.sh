#!/bin/bash
### AE
python train_ae.py --latent_dim 256 --epochs 100 --network VAE --instance homo_2 --lr1 1e-3
python train_ae.py --latent_dim 256 --epochs 100 --network VAE --instance homo_2 --lr1 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --network VAE --instance homo_2 --lr1 1e-5

python train_ae.py --latent_dim 128 --epochs 100 --network VAE --instance homo_2 --lr1 1e-3
python train_ae.py --latent_dim 128 --epochs 100 --network VAE --instance homo_2 --lr1 1e-4
python train_ae.py --latent_dim 128 --epochs 100 --network VAE --instance homo_2 --lr1 1e-5

python train_ae.py --latent_dim 64 --epochs 100 --network VAE --instance homo_2 --lr1 1e-3
python train_ae.py --latent_dim 64 --epochs 100 --network VAE --instance homo_2 --lr1 1e-4
python train_ae.py --latent_dim 64 --epochs 100 --network VAE --instance homo_2 --lr1 1e-5

: '
### AE reward
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward -lr3 1e-5--temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6
-lr3 1e-5-lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e4 --temperature_2 1e6 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6


#########

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward --temperature_1 1e3 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 0 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-5 --lr2 1e-6 --lr3 1e-6


python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-4 --lr2 1e-6 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-5 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-4 --lr3 1e-6

python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-4
python train_ae.py --latent_dim 256 --epochs 100 --instance hetero_1 --network AE_Reward 1e4 --temperature_2 1e5 --normalize  --weight_decay 1e-5 --lr1 1e-6 --lr2 1e-6 --lr3 1e-6

'