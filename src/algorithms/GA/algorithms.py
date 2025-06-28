import os
import sys
from numpy import genfromtxt
from utils import write_firewall_file, erase_firebreaks
sys.path.append("../../")
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import codecs
from torchvision.transforms import Normalize
import time
from numpy import linalg as LA
import random
from torch import nn


class Abstract_Genetic_Algorithm:
    """
    Super class of all genetic algorithms to be implemented.
    """
    def __init__(self, model, instance = "homo_2") -> None:
        """
        Initializes class:
            - setups de model
            - setups path to data and loads it
        """
        self.model = model
        self.model.eval()
        # Placeholders
        self.finetune = False
        self.lr = None
        self.epochs = None
        self.root = f"../../../data/complete_random/{instance}/"
        with open(self.root+"Sub20x20_full_grid.pkl", 'rb') as f:
            self.data = pickle.load(f)
        self.population = []
        self.valuations = []
        # storing rewards seperately
        self.rewards = []
        for i in range(len(self.data)):
            self.rewards.append(self.data[i][1])
        
        # Directory with landscape information
        self.landscape_dir = f"{self.root}/Sub20x20"

        # Loads elevation .asc into a numpy array
        with codecs.open(f'{self.landscape_dir}/elevation.asc', encoding='utf-8-sig', ) as f:
            line = "_"
            elevation = []
            while line:
                line = f.readline()
                line_list = line.split()
                if len(line_list) > 2:
                    elevation.append([float(i) for i in line_list])
        elevation = np.array(elevation)

        # Loads slope .asc into a numpy array
        with codecs.open(f'{self.landscape_dir}/slope.asc', encoding='utf-8-sig', ) as f:
            line = "_"
            slope = []
            while line:
                line = f.readline()
                line_list = line.split()
                if len(line_list) > 2:
                    slope.append([float(i) for i in line_list])
        slope = np.array(slope)

        # Loads elevation .saz into a numpy array
        with codecs.open(f'{self.landscape_dir}/saz.asc', encoding='utf-8-sig', ) as f:
            line = "_"
            saz = []
            while line:
                line = f.readline()
                line_list = line.split()
                if len(line_list) > 2:
                    saz.append([float(i) for i in line_list])
        saz = np.array(saz)

        # Stacks array into a tensor, generating a landscape tensor
        self.landscape = torch.from_numpy(np.stack([elevation, slope, saz]))
        # We compute means + std per channel to normalize
        means = torch.mean(self.landscape, dim=(1,2))
        stds = torch.std(self.landscape, dim=(1,2))
        norm = Normalize(means, stds)
        # Normalizes landscape
        self.landscape = norm(self.landscape)

    def initialize_population(self, initial_population = 0.01):
        """
        Initialized population, considering the best initial_population * 100 % of solutions and initial_population / 2 * 100 % of random 
        solutions.
        """
        # Getting best solutions
        self.data.sort(key=lambda x: x[1])
        self.population = self.data[len(self.data)- int(len(self.data) * initial_population) - 1:len(self.data) - 1]
        self.population.reverse()
        self.valuations = [i[1] for i in self.population]
        self.population = [self.transform(x) for x in self.population]
        # Adding random solutions
        rest = [i for i in range(0, len(self.data) - int(len(self.data) * initial_population) - 1)]
        sample = np.random.choice(rest, int(len(self.data) * (initial_population)), replace=False)
        for i in sample:
            self.population.append(self.transform(self.data[i]))
            self.valuations.append(self.data[i][1])
        print(len(self.population))

    def selection(self, population):
        pass

    def calc_fitness(self, embedding, n_sims = 50):
        """
        Calculates the average number of burned cells of embedding's associated
        solution.
        """

        solution = self.model.decode(embedding[0])
        _, indices = torch.topk(solution.flatten(), 20)
        indices = np.unravel_index(indices, (20, 20))
        matrix = torch.zeros((20, 20))
        matrix[indices] = 1.
        assert(matrix.sum().item() == 20)
        write_firewall_file(matrix * -1.)
        n_weathers = len([i for i in os.listdir(self.root+"Sub20x20/Weathers/") if i.endswith('.csv')])-2
        exec_str = f"../eval/C2F-W/Cell2FireC/Cell2Fire --input-instance-folder ../../../data/complete_random/homo_2/Sub20x20/ --output-folder ../eval/results/ --sim-years 1 --nsims {n_sims}--Fire-Period-Length 1.0 --output-messages --ROS-CV 2.0 --seed 123 --weather random --ignitions --IgnitionRad 4 --sim C --final-grid --nweathers {n_weathers} --FirebreakCells ../eval/harvested/HarvestedCells.csv"
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
        erase_firebreaks()
        return 1 + ((reward/n_sims) / 400)
    
    def transform(self, x):
        pass
    
    def fitness_func(self):
        pass

    def compute_similarity(self, embedding, population):
        pass

    def mutation(self, embedding):
        pass

    def cross_over(self, embedding_1, embedding_2):
        pass

    def stop_criteria(self):
        pass

    def train(self, n_iter = 1000, n_repeats=1):
        print("--------------Training started------------------")
        best = {}
        avg = {}
        for j in tqdm(range(n_repeats)):
            self.initialize_population(self.initial_population)
            best[j] = []
            avg[j] = []
            for i in tqdm(range(n_iter)):
                self.population_cross_over()
                self.population_mutation()
                if self.finetune:
                    self.fine_tune()
                self.selection()
                if self.stop_criteria():
                    break
                max_ = max(self.valuations)
                best[j].append(max_)
                avg[j].append(sum(self.valuations)/len(self.valuations))
                print(f"Current avg. score: {sum(self.valuations)/len(self.valuations)}, max valuation: {max_}")
        print("--------------Training stoped------------------")
        with open(f'results/best__{self.name}_{n_iter}_{self.params}.json', 'w') as f:
            json.dump(best, f)
        with open(f'results/avg__{self.name}_{n_iter}_{self.params}.json', 'w') as f:
           json.dump(avg, f)
        x = [i for i in range(n_iter)]
        best_mean = [0 for i in range(n_iter)]
        avg_mean = [0 for i in range(n_iter)]
        for j in range(n_repeats):
            for i in range(n_iter):
                best_mean[i] += best[j][i]
                avg_mean[i] += avg[j][i]
        best_mean = list(map(lambda x: x/n_repeats, best_mean))
        avg_mean = list(map(lambda x: x/n_repeats, avg_mean))
        best_std = [0 for i in range(n_iter)]
        avg_std = [0 for i in range(n_iter)]
        for i in range(n_iter):
            best_std[i] = sum([(best_mean[i] - best[j][i]) ** 2 / n_repeats  for j in range(n_repeats)]) ** 0.5
            avg_std[i] = sum([(avg_mean[i] - avg[j][i]) ** 2 / n_repeats  for j in range(n_repeats)]) ** 0.5
        print(best_mean, best_std)
        plt.plot(x, best_mean, lw=2, label="Best solution", color="blue")
        plt.fill_between(x, [best_mean[i] + best_std[i] for i in range(n_iter)], [best_mean[i] - best_std[i] for i in range(n_iter)], facecolor='blue', alpha=0.2)
        plt.plot(x, avg_mean, lw=2, label="Avg solution", color="red")
        plt.fill_between(x, [avg_mean[i] + avg_std[i] for i in range(n_iter)], [avg_mean[i] - avg_std[i] for i in range(n_iter)], facecolor='red', alpha=0.2)
        plt.legend()
        plt.savefig(f'results/plot__{self.name}_{n_iter}_{self.params}.png')
        plt.close()
        best = self.get_best()
        plt.imshow(best)
        plt.savefig(f"results/best_{self.name}_{n_iter}_{self.params}.png")
        plt.close()
        write_firewall_file(best * -1.)
        n_weathers = len([i for i in os.listdir(self.root+"Sub20x20/Weathers/") if i.endswith('.csv')])-2
        exec_str = f"../eval/C2F-W/Cell2FireC/Cell2Fire --input-instance-folder ../../../data/complete_random/homo_2/Sub20x20/ --output-folder ../eval/results/ --sim-years 1 --nsims 50 --Fire-Period-Length 1.0 --output-messages --ROS-CV 2.0 --seed 123 --weather random --ignitions --IgnitionRad 4 --sim C --final-grid --nweathers {n_weathers} --FirebreakCells ../eval/harvested/HarvestedCells.csv"
        os.system(exec_str + " >/dev/null 2>&1")
        base_directory = f"../eval/results/Grids/Grids"
        burn_probability = np.zeros((20, 20))
        reward = 0
        for j in range(1, 50+1):
            dir = f"{base_directory}{str(j)}/"
            files = os.listdir(dir)
            my_data = genfromtxt(dir+files[-1], delimiter=',')
            for cell in my_data.flatten():
                if cell == 1:
                    reward-= 1
            my_data = np.where(my_data == -1., 0, my_data )
            burn_probability += my_data
        burn_probability = burn_probability / 50
        plt.imshow(burn_probability, cmap="Reds")
        plt.colorbar()
        plt.savefig(f"results/bp_{self.name}_{n_iter}_{self.params}.png")

    def fine_tune(self):
        """
        Fine-tunes the model with the current population.
        """
        print("--------------Fine-tuning started------------------")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for i in tqdm(range(self.epochs)):
            for embedding in self.population:
                optimizer.zero_grad()
                mu, sigma = embedding
                loss = self.model.loss(mu, sigma)
                loss.backward()
                optimizer.step()
        print("--------------Fine-tuning stoped------------------")

        

class Vainilla_GA(Abstract_Genetic_Algorithm):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate=0.2, population_size=50,initial_population = 0.01) -> None:
        super().__init__(model, instance)

        self.sim_meassure = LA.norm
        self.name = "VANILLA_GA"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population

    def calc_fitness(self, solution, n_sims = 50):
        """
        Calculates the average number of burned cells of embedding's associated
        solution.
        """
        assert(solution.sum() == 20)
        write_firewall_file(solution * -1)
        n_weathers = len([i for i in os.listdir(self.root+"Sub20x20/Weathers/") if i.endswith('.csv')])-2
        exec_str = f"../eval/C2F-W/Cell2FireC/Cell2Fire --input-instance-folder ../../../data/complete_random/homo_2/Sub20x20/ --output-folder ../eval/results/ --sim-years 1 --nsims {n_sims}--Fire-Period-Length 1.0 --output-messages --ROS-CV 2.0 --seed 123 --weather random --ignitions --IgnitionRad 4 --sim C --final-grid --nweathers {n_weathers} --FirebreakCells ../eval/harvested/HarvestedCells.csv"
        os.system(exec_str + " >/dev/null 2>&1")
        reward = 0
        base_directory = f"../eval/results/Grids/Grids"
        for j in range(1, n_sims+1):
            dir = f"{base_directory}{str(j)}/"
            files = os.listdir(dir)
            my_data = genfromtxt(dir+files[-1], delimiter=',')
            # Burned cells are counted and turned into negative rewards
            for cell in my_data.flatten():
                if cell == 1.0:
                    reward-= 1
        return 1 + ((reward/n_sims) / 400)

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
            combined = [self.alpha * fitness[i] + (1-self.alpha) * self.compute_similarity(self.population[i], selected) / 1000 for i in range(len(self.population))]
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
                similarity += 1 - np.mean(matrix == i) / 400
            return similarity.item()/len(population)
        else:
            return 0
    

    def indiv_mutation(self, matrix):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        n, m = matrix.shape
        ones = []
        zeros = []
        for i in range(n):
            for j in range(m):
                if matrix[i,j]:
                    ones.append((i,j))
                else:
                    zeros.append((i,j))
        a = random.choice(ones)
        b = random.choice(zeros)
        mutation = matrix
        mutation[a] = 0.
        mutation[b] = 1.
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
        ones_a = []
        zeros_a = []
        ones_b = []
        zeros_b = []
        for i in range(n):
            for j in range(m):
                if matrix_1[i,j]:
                    ones_a.append((i,j))
                else:
                    zeros_a.append((i,j))
                if matrix_2[i,j]:
                    ones_b.append((i,j))
                else:
                    zeros_b.append((i,j))
        indices = list(set(ones_a + ones_b))
        cross_over = np.zeros((20, 20))
        for i in range(20):
            num = random.randint(0, len(indices)-1)
            idx = indices.pop(num)
            cross_over[idx] = 1.
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
        index_max = max(range(len(self.valuations)), key=self.valuations.__getitem__)
        return self.population[index_max]
    
class Variational_GA(Abstract_Genetic_Algorithm):
    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1) -> None:
        super().__init__(model, instance)
        self.sim_meassure = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.name = "VA_GA"
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.initial_population = initial_population
        self.params = f"alpha={self.alpha}_mutation_rate={self.mutation_rate}_population_size={self.population_size}_initial_population={self.initial_population}"
        self.lr = lr
        self.epochs = epochs

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
            stacked = torch.stack(list(zip(*population))[0]).squeeze(1)
            similarity = self.sim_meassure(embedding[0], stacked) 
            return similarity.sum().item()/len(population)
        else:
            return 1
    
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
        index_max = max(range(len(self.valuations)), key=self.valuations.__getitem__)
        mu, _ = self.population[index_max]
        solution = self.model.decode(mu)
        _, indices = torch.topk(solution.flatten(), 20)
        indices = np.unravel_index(indices, (20, 20))
        matrix = torch.zeros((20, 20))
        matrix[indices] = 1.
        assert(matrix.sum().item() == 20)
        return matrix

class Variational_GA_V1(Variational_GA):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1) -> None:
        # super().__init__(model, instance)
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_V1"

    
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
        t = time.process_time()
        temp = self.population.copy()
        for i in self.population:
            prob = np.random.uniform()
            if prob <= self.mutation_rate:
                temp.append(self.indiv_mutation(i))
        self.population = temp
        
    

class Variational_GA_V2(Variational_GA):

    def __init__(self, model, instance="homo_2", alpha = 0.5, mutation_rate = 0.2, population_size=50, initial_population = 0.01, lr=1e-5, epochs=1) -> None:
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_V2"

    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        mu = self.model.latent_sample(mu, sigma)
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
    

    

class Variational_GA_V1_CCVAE(Variational_GA_V1):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1) -> None:
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_V1_CCVAE"


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



class Variational_GA_V2_CCVAE(Variational_GA_V2):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1) -> None:
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_V2_CCVAE"

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

    


class Variational_GA_CD_CCVAE(Variational_GA_V1_CCVAE):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1, cond_thresh=0.75) -> None:
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_CD_CCVAE"
        self.threshold = cond_thresh
        self.params += f"_cond_thresh={cond_thresh}"


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


class Variational_GA_GD_CCVAE(Variational_GA_V1_CCVAE):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1, gradient_step=2) -> None:
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_GD_CCVAE"
        self.gradient_step = gradient_step
        self.params += f"_gradient_step={gradient_step}"


    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        latent = torch.tensor(mu[0, 128:]).clone().detach().unsqueeze(0).requires_grad_(True)
        latent_fixed = torch.tensor(mu[0, :128]).clone().detach().unsqueeze(0)
        self.optimizer = torch.optim.Adam([latent], lr=1e-1)
        for i in range(self.gradient_step):
            full_latent = torch.cat([latent_fixed, latent], dim=1)
            self.optimizer.zero_grad()
            loss = -self.model.predict_burned(full_latent)
            loss.backward()
            self.optimizer.step()
        mu = full_latent.detach()
        return (mu, sigma)


class Variational_GA_GD_V2_CCVAE(Variational_GA_V1_CCVAE):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1, gradient_step=2) -> None:
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_GD_V2_CCVAE"
        self.gradient_step = gradient_step
        self.params += f"_gradient_step={gradient_step}"


    def indiv_mutation(self, embedding):
        """
        Generates a mutation by sampling from N(mu, sigma)
        """
        mu, sigma = embedding
        latent = torch.tensor(mu[0, 128:]).clone().detach().unsqueeze(0).requires_grad_(True)
        latent_fixed = torch.tensor(mu[0, :128]).clone().detach().unsqueeze(0)
        self.optimizer = torch.optim.Adam([latent], lr=1e-1)
        for i in range(self.gradient_step):
            full_latent = torch.cat([latent_fixed, latent], dim=1)
            self.optimizer.zero_grad()
            loss = -self.model.predict_burned(full_latent)
            loss.backward()
            self.optimizer.step()
        mu, sigma = self.transform(self.model.decode(full_latent).detach()[0])
        return (mu, sigma)


class Variational_GA_MD_CCVAE(Variational_GA_V1_CCVAE):

    def __init__(self, model, instance="homo_2", alpha=0.5, mutation_rate = 0.2, population_size=50, initial_population=0.01, lr=1e-5, epochs=1) -> None:
        super().__init__(model, instance, alpha, mutation_rate, population_size, initial_population, lr, epochs)
        self.name = "VA_GA_MD_CCVAE"

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

    
    