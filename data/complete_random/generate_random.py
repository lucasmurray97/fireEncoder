import numpy as np
import random
import os
import sys
import csv
import pickle
import shutil
from tqdm import tqdm
from numpy import genfromtxt

seed = random.randint(0, 10000)
absolute_path = os.path.dirname(__file__)
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint(): 
    sys.stdout = sys.__stdout__

def erase_firebreaks():
    header = ['Year Number','Cell Numbers']
    path = f"{absolute_path}/homo_2/harvested/HarvestedCells.csv"
    # We empty out the firebreaks file
    with open(path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

def write_firebreaks(solution):
    
    """Function that writes the final state in the format delimited for firewalls in a file called HarvestedCells.csv"""
    absolute_path = os.path.dirname(__file__)
    try:
        shutil.rmtree(f'{absolute_path}/homo_2/results/Grids/')
    except:
        pass
    i = 1
    firewalls = [1]
    for cell in solution.flatten():
        if cell == -1:
            firewalls.append(i)
        i+=1
    header = ['Year Number','Cell Numbers']
    with open(f'{absolute_path}/homo_2/harvested/HarvestedCells.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(firewalls)
    return

def generate_reward(n_sims=100):
    """
    Calculates the average number of burned cells of embedding's associated
    solution.
    """
    n_weathers = len([i for i in os.listdir("homo_2/Sub20x20/Weathers/") if i.endswith('.csv')])-2
    exec_str = f"../../src/algorithms/eval/C2F-W/Cell2FireC/Cell2Fire --input-instance-folder ./homo_2/Sub20x20/ --output-folder ./homo_2/results/ --sim-years 1 --nsims {n_sims}--Fire-Period-Length 1.0 --output-messages --ROS-CV 1.0 --seed 123 --weather random --ignitions --IgnitionRad 4 --sim C --final-grid --nweathers {n_weathers} --FirebreakCells ./homo_2/harvested/HarvestedCells.csv"
    os.system(exec_str + " >/dev/null 2>&1")
    reward = 0
    base_directory = f"./homo_2/results/Grids/Grids"
    for j in range(1, n_sims+1):
        dir = f"{base_directory}{str(j)}/"
        files = os.listdir(dir)
        my_data = genfromtxt(dir+files[-1], delimiter=',')
        # Burned cells are counted and turned into negative rewards
        for cell in my_data.flatten():
            if cell == 1:
                reward-= 1
    return reward/n_sims

def generate_complete_random(n_sims = 10, size = 20):
    erase_firebreaks()
    n_cells = size**2
    n_firebreaks = int((n_cells)*0.05)
    available_cells = [i for i in range(n_cells)]
    firebreaks = np.random.choice(available_cells, size=n_firebreaks, replace=False)
    state = np.zeros(n_cells)
    state[firebreaks] = -1
    write_firebreaks(state)
    state = state.reshape(size,size)
    reward = generate_reward(n_sims)
    return state, reward

def generate_solutions_complete(observations,n_sims = 100):
    data = []
    for i in tqdm(range(observations)):
        state, evaluation = generate_complete_random(n_sims)
        data.append([state, evaluation])
    try:
        os.makedirs(f"{absolute_path}/homo_2/solutions/")
    except OSError as error:  
        print(error)
    with open(f"{absolute_path}/homo_2/solutions/Sub20x20_full_grid.pkl", "wb+") as write_file:
            pickle.dump(data, write_file)
    file = open(f"{absolute_path}/homo_2/solutions/Sub20x20_full_grid.pkl", 'rb')    
    n_r = pickle.load(file)
    print(len(n_r))
if __name__ == "__main__":
    generate_solutions_complete(50000)

