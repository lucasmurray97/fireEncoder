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

class Variational_GA_MD_CCVAE(Variational_GA_V1_CCVAE):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01) -> None:
        super().__init__(model, instance)
        self.name = "VA_GA_MD_CCVAE"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population
        self.argmax = np.argmax(self.rewards)
        self.argmin = np.argmin(self.rewards)
        self.max = self.transform(self.data[self.argmax])[0]
        self.min = self.transform(self.data[self.argmin])[0]
        # Compute direction between max and min
        self.max_dir = self.max - self.min
        self.max_dir /= torch.norm(self.max_dir)
        


    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        steps = 10
        mu += self.max_dir * 10 
        return (mu, sigma)

    