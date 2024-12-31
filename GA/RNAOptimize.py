import random
import numpy as np
from deap import base, creator, tools, algorithms

# フィットネスの定義（最大化の場合）
creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 個体生成とツールボックスの設定
toolbox = base.Toolbox()

def generate_individual():
    length = 50
    return [random.choice(['A', 'T', 'C', 'G']) for _ in range(length)]

toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 目的関数を定義
from modules import TE_predictor
from modules import Deg_predictor
TEpred = TE_predictor.predictor()
Degpred = Deg_predictor.predictor()
def evaluate(individual):
    seq = ''.join(individual)

    x = TEpred.predict(seq)  # 第一目的関数
    y = Deg_predictor.score_sum(Degpred.predict(seq))  # 第二目的関数
    return x, y

toolbox.register("evaluate", evaluate)

# 選択（NSGA-3）
toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(nobj=2, p=12))

# 交叉、突然変異、個体生成
def mutate(individual):
    idx = random.randint(0, len(individual)-1)
    individual[idx] = random.choice(['A', 'T', 'C', 'G'])
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("map", map)

# =====================
# メイン関数
# =====================
def main(seed=None):
    random.seed(seed)
    
    # 初期集団生成
    MU = 1000  # 集団サイズ
    NGEN = 400  # 世代数
    CXPB = 0.95  # 交叉確率
    pop = toolbox.population(n=MU)

    # 統計情報の収集用ツール
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # 最適化開始
    algorithms.eaMuPlusLambda(
        pop, toolbox, mu=MU, lambda_=MU, cxpb=CXPB, mutpb=1.0-CXPB,
        ngen=NGEN, stats=stats, halloffame=None, verbose=True
    )

    return pop

if __name__ == "__main__":
    result = main(seed=0)

    # 最終世代の個体を表示
    import pandas as pd
    df = pd.DataFrame(columns=["TE", "Deg", "seq", "cnt"])
    st = set()
    for indiv in result:
        x, y = indiv.fitness.values
        seq = ''.join(indiv)
        if seq in st:
            df.loc[df[df["seq"] == seq].index, "cnt"] += 1
        
        else:
            df.loc[len(df)] = [x, y, seq, 1]


    df.to_csv("/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/GA/result.csv", index=False)    
    print(df)

