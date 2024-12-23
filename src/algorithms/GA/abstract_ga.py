import os
import sys
from numpy import genfromtxt
from utils import write_firewall_file, erase_firebreaks
sys.path.append("../../")
from networks.vae import VAE
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import codecs
from torchvision.transforms import Normalize
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
        exec_str = f"../eval/C2F-W/Cell2FireC/Cell2Fire --input-instance-folder ../../../data/complete_random/homo_2/Sub20x20/ --output-folder ../eval/results/ --sim-years 1 --nsims 50 --Fire-Period-Length 1.0 --output-messages --ROS-CV 0.0 --seed 123 --weather random --ignitions --IgnitionRad 4 --sim C --final-grid --nweathers {n_weathers} --FirebreakCells ../eval/harvested/HarvestedCells.csv"
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

        