import random
import numpy as np
import optuna
from deap import base, creator, tools, algorithms
from modules import TE_predictor, Deg_predictor

# =====================
# 遺伝的アルゴリズムの設定
# =====================
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_individual():
    length = 50
    return [random.choice(['A', 'T', 'C', 'G']) for _ in range(length)]

toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 目的関数
TEpred = TE_predictor.predictor()
Degpred = Deg_predictor.predictor()

def evaluate(individual):
    seq = ''.join(individual)
    x = TEpred.predict(seq)  # 第一目的関数
    y = Deg_predictor.score_sum(Degpred.predict(seq))  # 第二目的関数
    return x, y

toolbox.register("evaluate", evaluate)

# 選択
toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(nobj=2, p=12))

def mutate(individual):
    idx = random.randint(0, len(individual) - 1)
    individual[idx] = random.choice(['A', 'T', 'C', 'G'])
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("map", map)

def run_ga(mu, ngen, cxpb, seed=None):
    random.seed(seed)
    pop = toolbox.population(n=mu)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(
        pop, toolbox, mu=mu, lambda_=mu, cxpb=cxpb, mutpb=1.0 - cxpb,
        ngen=ngen, stats=stats, halloffame=None, verbose=False
    )

    # 最良個体の評価値を返す
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind

# =====================
# Optunaによる多目的最適化
# =====================
def objective(trial):
    # ハイパーパラメータのサンプリング
    mu = trial.suggest_int("mu", 100, 1000, step=100)  # 集団サイズ
    ngen = trial.suggest_int("ngen", 50, 800, step=50)  # 世代数
    cxpb = trial.suggest_float("cxpb", 0.5, 1.0, step=0.05)  # 交叉確率
    

    # 遺伝的アルゴリズムの実行
    best_ind = run_ga(mu, ngen, cxpb, seed=0)
    fitness = best_ind.fitness.values

    # 第一目的関数と第二目的関数を返す
    return fitness[0], fitness[1]

def main():
    # 多目的最適化の設定
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    for i, trial in enumerate(study.best_trials):
        print(f"  Trial {i + 1}:")
        print(f"    Values: {trial.values}")
        print("    Params:")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")

    # 最適なハイパーパラメータでGAを実行
    best_trial = study.best_trials[0]
    print("\nRunning GA with best parameters...")
    result = run_ga(
        mu=best_trial.params["mu"],
        ngen=best_trial.params["ngen"],
        cxpb=best_trial.params["cxpb"],
        mutpb=best_trial.params["mutpb"],
        seed=0
    )

    print("Best individual:")
    print(f"  TE: {result.fitness.values[0]}")
    print(f"  Deg: {result.fitness.values[1]}")
    print(f"  Sequence: {''.join(result)}")

if __name__ == "__main__":
    main()
