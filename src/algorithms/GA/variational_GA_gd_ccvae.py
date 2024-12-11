import os
import sys
from numpy import genfromtxt
from utils import write_firewall_file
sys.path.append("../../")
from networks.vae import VAE
import torch
from torch import nn
import pickle
from abstract_ga import Abstract_Genetic_Algorithm
from variational_GA import Variational_GA_V1
from variational_GA_ccvae import Variational_GA_V1_CCVAE
import random
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import time

class Variational_GA_GD_CCVAE(Variational_GA_V1_CCVAE):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01) -> None:
        super().__init__(model, instance)
        self.name = "VA_GA_GD_CCVAE"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population


    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        latent = torch.tensor(mu[0, 128:]).clone().detach().unsqueeze(0).requires_grad_(True)
        latent_fixed = torch.tensor(mu[0, :128]).clone().detach().unsqueeze(0)
        self.optimizer = torch.optim.Adam([latent], lr=1e-1)
        steps = 20
        for i in range(20):
            full_latent = torch.cat([latent_fixed, latent], dim=1)
            self.optimizer.zero_grad()
            loss = -self.model.predict_burned(full_latent)
            loss.backward()
            self.optimizer.step()
        mu = full_latent.detach()
        return (mu, sigma)

    