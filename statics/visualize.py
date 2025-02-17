import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def vis_diag(jobid):
    with open('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/logbook/logbook' + str(jobid) + '.pkl', 'rb') as f:
        logbook = pickle.load(f)

    fxy = logbook.select("avg")
    gen = logbook.select("gen")
    
    fx = [x[0] for x in fxy]
    fy = [x[1] for x in fxy]

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
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')
    plt.savefig(f"/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/statics/nsga3_two/GARNA" + str(jobid) + ".png", dpi=300, bbox_inches='tight')


def vis_diag_all(joblist, algo):
    palette = sns.color_palette("bright", len(joblist))
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    for i, jobid in enumerate(joblist):
        print(f'i: {i}, jobid: {jobid}')
        with open('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/logbook/logbook' + str(jobid) + '.pkl', 'rb') as f:
            logbook = pickle.load(f)

        fxy = logbook.select("avg")
        gen = logbook.select("gen")
        
        fx = [x[0] for x in fxy]
        fy = [x[1] for x in fxy]

        # print(f"color: {palette[i]}")
        sns.lineplot(x=gen, y=fx, ax=ax1, label=f"TE avg{i}", color=palette[i], legend=False)
        sns.lineplot(x=gen, y=fy, ax=ax2, label=f"DEG avg{i}", color=palette[i], legend=False)

        fx = fx[::100]
        fy = fy[::100]
        gen = gen[::100]
        sns.scatterplot(x=gen, y=fx, ax=ax1, color=palette[i], legend=False, marker='o', s=80)


    ax1.set_xlabel('Generation', fontsize=14)
    ax1.set_ylabel('TE', color='blue', fontsize=14)
    ax1.set_ylim(-0.7, 2.5)
    ax2.set_ylabel('DEG', color='red', fontsize=14)
    ax2.set_ylim(5.0, 24)

    handles1, labels1 = ax1.get_legend_handles_labels()
    legend_elem = [
        plt.Line2D([0], [0], color='black', linestyle='-', marker='o', markersize=5, label='TE avg'),
        plt.Line2D([0], [0], color='black', linestyle='-', label='DEG avg')
    ]

    if(algo == 'nsga2_unif'):
        ax1.legend(handles1 + legend_elem, ['ex1', 'ex2', 'ex3', 'ex4', 'TE avg', 'DEG avg'], loc='center right')
    elif(algo == 'nsga2_two'):
        ax1.legend(handles1 + legend_elem, ['ex1', 'ex2', 'ex3', 'ex4', 'TE avg', 'DEG avg'], loc='upper right')
    elif(algo == 'nsga3_unif'):
        ax1.legend(handles1 + legend_elem, ['ex1', 'ex2', 'ex3', 'ex4', 'TE avg', 'DEG avg'], loc='lower right', bbox_to_anchor=(1.0, 0.1))
    elif(algo == 'nsga3_two'):
        ax1.legend(handles1 + legend_elem, ['ex1', 'ex2', 'ex3', 'ex4', 'TE avg', 'DEG avg'], loc='upper right')


    plt.savefig(f"/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/statics/"+algo+"/" + algo + "_diag.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    import indiv_check as ic
    joblist = list(ic.nsga3_unif.keys())
    vis_diag_all(joblist, "nsga3_unif")