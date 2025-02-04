import random
import numpy as np
from deap import base, creator, tools, algorithms

# Types
creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0))
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
    individual.x_value = x
    return (y, 0)

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

mate_f = "cxTwoPoint"
# mate_f = "cxUniform"
if(mate_f == "cxTwoPoint"):
    print(f"mate: cxTwoPoint")
    toolbox.register("mate", tools.cxTwoPoint)
else:
    print(f"mate: cxUniform (indpb=0.5)")
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

if __name__ == "__main__":
    random.seed(0)
    
    # 初期集団生成
    MU = 2000  # 集団サイズ
    NGEN = 1000  # 世代数
    CXPB = 0.75  # 交叉確率
    MUTPB = 0.25  # 突然変異確率
    pop = toolbox.population(n=MU)

    print(f"params: MU={MU}, NGEN={NGEN}, CXPB={CXPB}, MUTPB={MUTPB}")

    # 統計情報の収集用ツール
    stats = tools.Statistics(lambda ind: (ind.fitness.values[0], getattr(ind, "x_value", None)))
    stats.register("avg", lambda vals: (np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals if v[1] is not None])))
    stats.register("std", lambda vals: (np.std([v[0] for v in vals]), np.std([v[1] for v in vals if v[1] is not None])))
    stats.register("min", lambda vals: (np.min([v[0] for v in vals]), np.min([v[1] for v in vals if v[1] is not None])))
    stats.register("max", lambda vals: (np.max([v[0] for v in vals]), np.max([v[1] for v in vals if v[1] is not None])))


    print(creator.Individual.fitness.weights)

    # 最適化開始
    result, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=MU, lambda_=MU, cxpb=CXPB, mutpb=MUTPB,
        ngen=NGEN, stats=stats, halloffame=None, verbose=True
    )


    import pickle
    import os
    jobid = os.environ['SLURM_JOB_ID']
    with open('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/img/logbook' + jobid + '_Degonly.pkl', 'wb') as f:
        pickle.dump(logbook, f)
    
    # plot result
    gen = logbook.select("gen")

    fit_mins = logbook.select("min")
    fit_maxs = logbook.select("max")
    fit_avgs = logbook.select("avg")

    te_mins = [x[1] for x in fit_mins]
    te_avgs = [x[1] for x in fit_avgs]
    deg_maxs = [x[0] for x in fit_maxs]
    deg_avgs = [x[0] for x in fit_avgs]


    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(gen, te_mins, label='TE min', color='blue')
    ax1.plot(gen, te_avgs, label='TE avg', color='lightblue')
    ax2.plot(gen, deg_maxs, label='DEG max', color='red')
    ax2.plot(gen, deg_avgs, label='DEG avg', color='salmon')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('TE', color='blue')
    ax2.set_ylabel('DEG', color='red')
    
    # ax1とax2の凡例をまとめて出力
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='center right')

    
    plt.savefig('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/img/plot' + jobid + '_Degonly.png')

    seqs = ["".join(ind) for ind in pop]
    st = set(seqs)
    seqs = list(st)
    print(seqs)