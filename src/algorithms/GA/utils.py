import sys
import numpy as np
from numpy import genfromtxt
import csv  
import pprint
import os
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main
import os
import shutil
import random
def write_firewall_file(solution):
    
    """Function that writes the final state in the format delimited for firewalls in a file called HarvestedCells.csv"""
    absolute_path = os.path.dirname(__file__)
    try:
        shutil.rmtree(f'{absolute_path}/../eval/results/Grids/')
    except:
        pass
    i = 1
    firewalls = [1]
    for cell in solution.flatten():
        if cell == -1:
            firewalls.append(i)
        i+=1
    header = ['Year Number','Cell Numbers']
    with open(f'{absolute_path}/../eval/harvested/HarvestedCells.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(firewalls)
    return