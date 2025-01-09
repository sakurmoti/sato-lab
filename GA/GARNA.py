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

    # x = TEpred.predict(seq)  # 第一目的関数
    x=5
    y = Deg_predictor.score_sum(Degpred.predict(seq), label='reactivity')  # 第二目的関数
    return x, y

def mutate(individual):
    idx = random.randint(0, len(individual)-1)
    individual[idx] = random.choice(['A', 'T', 'C', 'G'])
    return individual,

toolbox.register("evaluate", evaluate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(nobj=2, p=12))
toolbox.register("mate", tools.cxTwoPoint)

if __name__ == "__main__":
    random.seed(0)
    
    # 初期集団生成
    MU = 1000  # 集団サイズ
    NGEN = 500  # 世代数
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

    record = stats.compile(pop)

    import pickle
    with open('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/img/logbook702.pkl', 'wb') as f:
        pickle.dump(logbook, f)

    # plot result
    gen = logbook.select("gen")

    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    te_mins = [x[0] for x in fit_mins]
    te_avgs = [x[0] for x in fit_avgs]
    deg_mins = [x[1] for x in fit_mins]
    deg_avgs = [x[1] for x in fit_avgs]


    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(gen, te_mins, label='TE min', color='blue')
    ax1.plot(gen, te_avgs, label='TE avg', color='lightblue')
    ax2.plot(gen, deg_mins, label='DEG min', color='red')
    ax2.plot(gen, deg_avgs, label='DEG avg', color='salmon')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('TE', color='blue')
    ax2.set_ylabel('DEG', color='red')
    
    # ax1とax2の凡例をまとめて出力
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='center right')

    
    plt.savefig('/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/deap/img/plot702.png')

    seqs = ["".join(ind) for ind in pop]
    print(seqs)