import os
import sys
from numpy import genfromtxt
from utils import write_firewall_file
sys.path.append("../../")
from networks.vae import VAE
import torch
import pickle
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
        self.root = f"../../../data/complete_random/{instance}/Sub20x20_full_grid.pkl"
        with open(self.root, 'rb') as f:
            self.data = pickle.load(f)
        self.population = []
        self.valuations = []

    def initialize_population(self, alpha = 0.01):
        """
        Initialized population, considering the best alpha * 100 % of solutions.
        """
        self.data.sort(key=lambda x: x[1])
        self.population = self.data[len(self.data)- int(len(self.data) * alpha) - 1:len(self.data) - 1]
        self.population.reverse()
        self.valuations = [i[1] for i in self.population]
        self.population = [(self.model.encode(torch.Tensor(x[0]).unsqueeze(0).unsqueeze(0))) for x in self.population]

    def selection(self, population):
        pass

    def calc_fitness(self, embedding, n_sims = 10):
        """
        Calculates the average number of burned cells of embedding's associated
        solution.
        """
        solution = self.model.decode(embedding[0])
        write_firewall_file((solution > 0.5) * -1)
        exec_str = f"../eval/C2F-W/Cell2FireC/Cell2Fire --input-instance-folder ../../../data/complete_random/homo_2/Sub20x20/ --output-folder ../eval/results/ --sim-years 1 --nsims {n_sims}--Fire-Period-Length 1.0 --output-messages --ROS-CV 1.0 --seed 123 --weather random --ignitions --IgnitionRad 4 --sim C --final-grid --nweather 359 --FirebreakCells ../eval/harvested/HarvestedCells.csv"
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
    
    def fitness_func(self):
        pass

    def compute_similarity(self, embedding, population):
        pass

    def mutation(self, embedding):
        pass

    def cross_over(self, embedding_1, embedding_2):
        pass

    def train(self):
        pass

    def stop_criteria(self):
        pass
