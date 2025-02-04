import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load logbook
jobid = 743
with open('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/img/logbook' + str(jobid) + '.pkl', 'rb') as f:
    logbook = pickle.load(f)


fxy = logbook.select("avg")
gen = logbook.select("gen")

fx = [x[0] for x in fxy]
fy = [x[1] for x in fxy]

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(gen, fx, label='TE avg', color='blue')
# ax2.plot(gen, fy, label='DEG avg', color='red')
# plt.xlabel("Generation")
# ax1.set_ylabel('TE', color='blue')
# ax2.set_ylabel('DEG', color='red')
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(handles1 + handles2, labels1 + labels2)
# plt.savefig(f"img/GARNA" + str(jobid) + ".png")


fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twinx()

sns.lineplot(x=gen, y=fx, ax=ax1, label='TE avg', color='blue', legend=False)
sns.lineplot(x=gen, y=fy, ax=ax2, label='DEG avg', color='red', legend=False)

ax1.set_xlabel('Generation', fontsize=14)
ax1.set_ylabel('TE', color='blue', fontsize=14)
ax1.set_ylim(-0.7, 2.5)
ax2.set_ylabel('DEG', color='red', fontsize=14)
ax2.set_ylim(5.0, 24)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2)
plt.savefig(f"img/GARNA" + str(jobid) + ".png", dpi=300, bbox_inches='tight')
