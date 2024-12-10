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
import random
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import time

class Variational_GA_V1_CCVAE(Variational_GA_V1):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01) -> None:
        super().__init__(model, instance)
        self.name = "VA_GA_V1_CCVAE"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population


    def transform(self, x):
        x = x[0][np.newaxis, :, :]
        x = np.concatenate([x, self.landscape], axis=0)
        return self.model.encode(torch.Tensor(x).unsqueeze(0))
                
    def retrieve_sigma(self, embedding):
        """
        Retrieves sigma associated with an embedding
        """
        sol = self.model.decode(embedding)
        sol = np.concatenate([sol, self.landscape], axis=0)
        return self.model.encode(sol)

    