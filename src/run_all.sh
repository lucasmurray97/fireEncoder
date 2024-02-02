#!/bin/bash

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6 --normalize  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6 --normalize --weight_decay 1e-1 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6 --normalize --weight_decay 1e-2 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6 --normalize --weight_decay 1e-3 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6  --weight_decay 0 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6  --weight_decay 1e-1 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6  --weight_decay 1e-2 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1e6  --weight_decay 1e-3 --lr1 1e-5 --lr2 1e-5 --lr3 1e-5