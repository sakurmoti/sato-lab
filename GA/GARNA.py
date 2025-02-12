import random
import numpy as np
from deap import base, creator, tools, algorithms

# Types
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize
def generate_individual():
    length = 50
    return [random.choice(['A', 'T', 'C', 'G']) for _ in range(length)]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Operators
from modules import TE_predictor
from modules import Deg_predictor
TEpred = TE_predictor.predictor()
Degpred = Deg_predictor.predictor()
def evaluate(individual):
    seq = ''.join(individual)

    x = TEpred.predict(seq)  # 第一目的関数
    y = Deg_predictor.score_sum(Degpred.predict(seq), label='deg_50C')  # 第二目的関数
    return x, y

def mutate(individual):
    indpb = 0.5 # 変異する遺伝子の割合
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(['A', 'T', 'C', 'G'])
    return individual,

toolbox.register("evaluate", evaluate)


print("mutation: mutate, indpb=0.5")
toolbox.register("mutate", mutate)


sel_f = "selNSGA3"
# sel_f = "selNSGA2"
if(sel_f == "selNSGA3"):
    print("selection: selNSGA3")
    toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(nobj=2, p=12))
else:
    print("selection: selNSGA2")
    toolbox.register("select", tools.selNSGA2)

# mate_f = "cxTwoPoint"
mate_f = "cxUniform"
if(mate_f == "cxTwoPoint"):
    print(f"mate: cxTwoPoint")
    toolbox.register("mate", tools.cxTwoPoint)
else:
    print(f"mate: cxUniform (indpb=0.5)")
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

def main(fold=None):
    random.seed(0)

    # 初期集団生成
    MU = 2000  # 集団サイズ
    NGEN = 1000  # 世代数
    CXPB = 0.75  # 交叉確率
    MUTPB = 0.25  # 突然変異確率
    pop = toolbox.population(n=MU)

    print(f"params: MU={MU}, NGEN={NGEN}, CXPB={CXPB}, MUTPB={MUTPB}")

    # 統計情報の収集用ツール
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # 最適化開始
    result, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=MU, lambda_=MU, cxpb=CXPB, mutpb=MUTPB,
        ngen=NGEN, stats=stats, halloffame=None, verbose=True
    )


    import pickle
    import os
    jobid = os.environ['SLURM_JOB_ID']
    path = '/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/logbook/logbook' + jobid
    if(fold is not None):
        path += f"_fold{fold}"
    path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(logbook, f)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
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
    ax1.legend(handles1 + handles2, labels1 + labels2)

    path = '/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/img/plot' + jobid
    if(fold is not None):
        path += f"_fold{fold}"
    path += '.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')

    seqs = ["".join(ind) for ind in pop]
    st = set(seqs)
    seqs = list(st)
    print(len(seqs))
    print(seqs)

if __name__ == "__main__":
    main()