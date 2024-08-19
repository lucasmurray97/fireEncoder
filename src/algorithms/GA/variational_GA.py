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
import random
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import time

class Variational_GA_V1(Abstract_Genetic_Algorithm):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01) -> None:
        super().__init__(model, instance)
        self.sim_meassure = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.name = "VA_GA_V1"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population

    def selection(self):
        """
        Selects population_size elements from current population by computing a score that ponderates
        fitness and diversity.
        """
        selected = []
        fitness = []
        scores = []
        chosen = self.population_size
        for i in range(len(self.population)):
            if i < len(self.valuations):
                fitness.append(self.valuations[i])
            else:
                fitness.append(self.calc_fitness(self.population[i]))
        index_max = max(range(len(fitness)), key=fitness.__getitem__)
        selected.append(self.population[index_max])
        scores.append(fitness[index_max])
        self.population.pop(index_max)
        first = fitness.pop(index_max)
        chosen -= 1
        self.valuations = [first]
        while(chosen):
            combined = [self.alpha * fitness[i] + (1-self.alpha) * self.compute_similarity(self.population[i], selected) / 100 for i in range(len(self.population))]
            index_max = max(range(len(combined)), key=combined.__getitem__)
            selected.append(self.population[index_max])
            self.population.pop(index_max)
            self.valuations.append(fitness.pop(index_max))
            scores.append(combined[index_max])
            chosen -= 1
        self.population = selected
        return selected


    def compute_similarity(self, embedding, population):
        """
        Computes the average cosine similarity between embedding and current selected population
        """
        similarity = 0
        if population:
            for i in population:
                similarity -=  self.sim_meassure(embedding[0], i[0])
            
            return similarity.item()/len(population)
        else:
            return 1
    

    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        _, dims = mu.shape
        chosen_dim = random.randrange(0, dims-1)
        std = sigma[0][chosen_dim].mul(0.5).exp_()
        eps = torch.normal(torch.zeros(std.shape))
        sample = mu[0][chosen_dim] + (eps * std)
        mu[0][chosen_dim] = sample
        return (mu, sigma)
    
    def population_mutation(self):
        """
        Generates mutations over whole population
        """
        temp = self.population.copy()
        for i in self.population:
            prob = np.random.uniform()
            if prob <= self.mutation_rate:
                temp.append(self.indiv_mutation(i))
        self.population = temp
    
    def indiv_cross_over(self, embedding_1, embedding_2):
        """
        Interpolates between embedding_1 and embedding_2 by a simple average
        """
        mu_1, sigma_1 =  embedding_1
        mu_2, sigma_2 = embedding_2
        interpolation = (mu_1 + mu_2) / 2
        sigma = (sigma_1 + sigma_2) / 2
        return (interpolation, sigma)

    def population_cross_over(self):
        """
        Applies indiv_cross_over to population
        """
        temp = self.population.copy()
        while(len(temp) > 1):
            parent_1 = temp.pop(random.randrange(len(temp)))
            parent_2 = temp.pop(random.randrange(len(temp)))
            offspring = self.indiv_cross_over(parent_1, parent_2)
            self.population.append(offspring)
        
    def transform(self, x):
        return self.model.encode(torch.Tensor(x[0]).unsqueeze(0).unsqueeze(0))
                
    def retrieve_sigma(self, embedding):
        """
        Retrieves sigma associated with an embedding
        """
        return self.model.encode(self.model.decode(embedding))

    def stop_criteria(self):
        return False
    
    def get_best(self):
        fitness = []
        for i in range(len(self.population)):
            if i < len(self.valuations):
                fitness.append(self.valuations[i])
            else:
                fitness.append(self.calc_fitness(self.population[i]))
        index_max = max(range(len(fitness)), key=fitness.__getitem__)
        print(self.model.decode(self.population[index_max][0]) > 0.5)
        return index_max
    