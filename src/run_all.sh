#!/bin/bash
### AE
python train_ae.py --latent_dim 256 --epochs 100 --network CCVAE --instance homo_2 --lr1 1e-5 --lr2 1e-5 --distribution_std 0.1 --sigmoid --variational_beta 0.1 --alpha 1e4 --data_version 1
python train_ae.py --latent_dim 256 --epochs 100 --network CCVAE --instance homo_2 --lr1 1e-5 --lr2 1e-5 --distribution_std 0.1 --sigmoid --variational_beta 0.1 --alpha 1e5 --data_version 1
python train_ae.py --latent_dim 256 --epochs 100 --network CCVAE --instance homo_2 --lr1 1e-5 --lr2 1e-5 --distribution_std 0.1 --sigmoid --variational_beta 0.1 --alpha 1e6 --data_version 1

python train_ae.py --latent_dim 256 --epochs 100 --network CCVAE --instance homo_2 --lr1 1e-5 --lr2 1e-6 --distribution_std 0.1 --sigmoid --variational_beta 0.1 --alpha 1e4 --data_version 1
python train_ae.py --latent_dim 256 --epochs 100 --network CCVAE --instance homo_2 --lr1 1e-5 --lr2 1e-6 --distribution_std 0.1 --sigmoid --variational_beta 0.1 --alpha 1e5 --data_version 1
python train_ae.py --latent_dim 256 --epochs 100 --network CCVAE --instance homo_2 --lr1 1e-5 --lr2 1e-6 --distribution_std 0.1 --sigmoid --variational_beta 0.1 --alpha 1e6 --data_version 1