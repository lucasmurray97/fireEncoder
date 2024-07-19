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
from numpy import linalg as LA

class Vainilla_GA(Abstract_Genetic_Algorithm):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate=0.2, population_size=50,initial_population = 0.01) -> None:
        super().__init__(model, instance)

        self.sim_meassure = LA.norm
        self.name = "VANILLA_GA"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population
    def calc_fitness(self, solution, n_sims = 10):
        """
        Calculates the average number of burned cells of embedding's associated
        solution.
        """
        write_firewall_file(solution * -1)
        n_weathers = len([i for i in os.listdir(self.root+"Sub20x20/Weathers/") if i.endswith('.csv')])-2
        exec_str = f"../eval/C2F-W/Cell2FireC/Cell2Fire --input-instance-folder ../../../data/complete_random/homo_2/Sub20x20/ --output-folder ../eval/results/ --sim-years 1 --nsims {n_sims}--Fire-Period-Length 1.0 --output-messages --ROS-CV 0.0 --seed 123 --weather random --ignitions --IgnitionRad 4 --sim C --final-grid --nweathers {n_weathers} --FirebreakCells ../eval/harvested/HarvestedCells.csv"
        os.system(exec_str + " >/dev/null 2>&1")
        reward = 0
        base_directory = f"../eval/results/Grids/Grids"
        for j in range(1, n_sims+1):
            dir = f"{base_directory}{str(j)}/"
            files = os.listdir(dir)
            my_data = genfromtxt(dir+files[-1], delimiter=',')
            # Burned cells are counted and turned into negative rewards
            for cell in my_data.flatten():
                if cell == 1:
                    reward-= 1
        return reward/n_sims

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
            combined = [self.alpha * fitness[i] / 100 + (1-self.alpha) * self.compute_similarity(self.population[i], selected) for i in range(len(self.population))]
            index_max = max(range(len(combined)), key=combined.__getitem__)
            selected.append(self.population[index_max])
            self.population.pop(index_max)
            self.valuations.append(fitness.pop(index_max))
            scores.append(combined[index_max])
            chosen -= 1
        self.population = selected
        return selected


    def compute_similarity(self, matrix, population):
        """
        Computes the average cosine similarity between matrix and current selected population
        """
        similarity = 0
        if population:
            for i in population:
                similarity += self.sim_meassure(matrix - i)
            return similarity.item()/len(population)
        else:
            return 0
    

    def indiv_mutation(self, matrix):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        n, m = matrix.shape
        available_cells_x = [x for x in range(n)]
        available_cells_y = [y for y in range(m)]
        i = np.random.choice(available_cells_x, size=1)
        j = np.random.choice(available_cells_y, size=1)
        mutation = matrix
        mutation[i,j] = not matrix[i,j]
        return mutation
    
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
    
    def indiv_cross_over(self, matrix_1, matrix_2):
        """
        Interpolates between matrix_1 and matrix_2 by a simple average
        """
        n, m = matrix_1.shape
        flat_m1 = matrix_1.flatten()
        flat_m2 = matrix_2.flatten() 
        index = np.random.choice([i for i in range(1,n)], size=1)[0]
        cross_over = np.zeros(n*m, dtype=int)
        cross_over[0:index-1] = flat_m1[0:index-1]
        cross_over[index:n-1] = flat_m2[index:n-1]
        cross_over = cross_over.reshape((n,m))
        return cross_over

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
        return x[0]
                

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
    