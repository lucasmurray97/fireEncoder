#!/bin/bash
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 0 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 0 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 0 --lr 1e-5


python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 1e-5 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 1e-5 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 1e-5 --lr 1e-5


python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 1e-7 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 1e-7 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1 --normalize --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10 --normalize --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100 --normalize --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000 --normalize --weight_decay 1e-7 --lr 1e-5


python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 0 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 0 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 0 --lr 1e-5


python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 1e-5 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 1e-5 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 1e-5 --lr 1e-5


python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 1e-7 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 1e-7 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1  --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10  --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100  --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000  --weight_decay 1e-7 --lr 1e-5