import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load logbook
with open('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/img/logbook635.pkl', 'rb') as f:
    logbook = pickle.load(f)


fxy = logbook.select("avg")

fig, ax = plt.subplots()
ax.plot(fxy)
ax.set_xlabel("Generation")
ax.set_ylabel("Average Fitness")
plt.savefig('generation_te_deg.png')
