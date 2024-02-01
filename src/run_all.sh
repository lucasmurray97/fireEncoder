#!/bin/bash
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 0 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 0 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 0 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 0 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 0 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 0 --lr 1e-5


python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 1e-5 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 1e-5 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 1e-5 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 1e-5 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 1e-5 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 1e-5 --lr 1e-5


python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 1e-7 --lr 0.0001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 1e-7 --lr 0.0001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 1e-7 --lr 0.001
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 1e-7 --lr 0.001

python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000  --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 1000000  --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 10000000  --weight_decay 1e-7 --lr 1e-5
python train_ae.py --latent_dim 256 --epochs 100 --network AE_Reward --temperature 100000000  --weight_decay 1e-7 --lr 1e-5