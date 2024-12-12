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

class Variational_GA_CD_CCVAE(Variational_GA_V1_CCVAE):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, cond_thresh=0.75) -> None:
        super().__init__(model, instance)
        self.name = "VA_GA_CD_CCVAE"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population
        self.threshold = cond_thresh
        self.params = f"alpha={self.alpha}_mutation_rate={self.mutation_rate}_population_size={self.population_size}_initial_population={self.initial_population}_threshold={self.threshold}"


    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        z = torch.randn(1,256)
        y = self.model.predict_burned(z)
        while y < self.threshold:
            z = torch.randn(1,256)
            y = self.model.predict_burned(z)
        sigma = self.retrieve_sigma(z)
        return (z, sigma)

    def retrieve_sigma(self, embedding):
        """
        Retrieves sigma associated with an embedding
        """
        x = self.model.decode(embedding).detach()[0]
        x = np.concatenate([x, self.landscape], axis=0).astype(np.float32)
        mu, sigma = self.model.encode(torch.from_numpy(x).unsqueeze(0))
        return sigma

    