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
import numpy as np
import matplotlib.pyplot as plt

class Variational_GA_V2(Abstract_Genetic_Algorithm):

    def __init__(self, model, instance="homo_2", mutation_rate = 0.2) -> None:
        super().__init__(model, instance)
        self.sim_meassure = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.name = "VA_GA_V2"
        self.mutation_rate = mutation_rate
    def selection(self, alpha = 0.5, population_size = 50):
        """
        Selects population_size elements from current population by computing a score that ponderates
        fitness and diversity.
        """
        selected = []
        fitness = []
        scores = []
        chosen = population_size
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
            combined = [alpha * fitness[i] + (1-alpha) * self.compute_similarity(self.population[i], selected) for i in range(len(self.population))]
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
                similarity += self.sim_meassure(embedding[0], i[0])
            return similarity.item()/len(population)
        else:
            return 0
    

    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        mu = self.model.latent_sample(mu, sigma)
        _, sigma = self.retrieve_sigma(mu)
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
        mu_1, _ =  embedding_1
        mu_2, _ = embedding_2
        interpolation = (mu_1 + mu_2) / 2
        _, sigma = self.retrieve_sigma(interpolation)
        return (interpolation, sigma)

    def population_cross_over(self):
        """
        Applies indiv_cross_over to population
        """
        temp = self.population.copy()
        while(temp):
            parent_1 = temp.pop(random.randrange(len(temp)))
            parent_2 = temp.pop(random.randrange(len(temp)))
            offspring = self.indiv_cross_over(parent_1, parent_2)
            self.population.append(offspring)
                
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
    
    def train(self, n_iter = 1000):
        print("--------------Training started------------------")
        best = []
        avg = []
        for _ in tqdm(range(n_iter)):
            self.population_cross_over()
            self.population_mutation()
            self.selection()
            if self.stop_criteria():
                break
            max_ = max(self.valuations)
            best.append(max_)
            avg.append(sum(self.valuations)/len(self.valuations))
            print(f"Current avg. score: {sum(self.valuations)/len(self.valuations)}, max valuation: {max_}")
        print("--------------Training stoped------------------")
        with open(f'results/best_{self.name}_{n_iter}.json', 'w') as f:
            json.dump(best, f)
        with open(f'results/avg_{self.name}_{n_iter}.json', 'w') as f:
           json.dump(avg, f)
        x = [i for i in range(len(best))]
        plt.plot(x, best, label="Best solution")
        plt.plot(x, avg, label="Population Average")
        plt.legend()
        plt.savefig(f'results/plot_{self.name}_{n_iter}.png')