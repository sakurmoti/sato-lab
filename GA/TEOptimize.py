import numpy as np
import random
import pandas as pd
from deap import base, creator, tools, algorithms
import sys
from modules import TE_predictor
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # 最大化問題として定義
creator.create("Individual", list, fitness=creator.FitnessMax) # 個体の定義

# toolbox.registerにより関数を登録する
toolbox = base.Toolbox()
toolbox.register("attr_rna", random.choice, ['A', 'T', 'C', 'G']) # 個体の取りうる値を決める関数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rna, n=100) # 個体を生成する関数
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # 集団を生成する関数

# 選択 := トーナメント選択
toolbox.register("select", tools.selTournament, tournsize=3)

# 交叉 := 2つの親個体をランダムに選び、ランダムな遺伝子座を選んで交換する
def cxRna(ind1, ind2):
    idx = random.randint(0, len(ind1)-1)
    ind1[idx], ind2[idx] = ind2[idx], ind1[idx]
    return ind1, ind2
toolbox.register("mate", cxRna)

# 突然変異 := 個体の遺伝子座をランダムに選び、ランダムな遺伝子に置き換える
def mutRna(individual):
    idx = random.randint(0, len(individual)-1)
    individual[idx] = random.choice(['A', 'T', 'C', 'G'])
    return individual,
toolbox.register("mutate", mutRna)

# TEの最大化を図る
model = TE_predictor.predictor()
def evaluate(individual):
    seq = ''.join(individual)
    # print(seq)
    score = model.predict(seq)

    return score,
    

    
toolbox.register("evaluate", evaluate)


def opts():
    import argparse
    parser = argparse.ArgumentParser(description='GA')
    parser.add_argument('--pop', type=int, default=300, help='population size')
    parser.add_argument('--ngen', type=int, default=200, help='number of generation')
    parser.add_argument('--cxpb', type=float, default=0.9, help='crossover probability')
    parser.add_argument('--mutpb', type=float, default=0.1, help='mutation probability')
    args = parser.parse_args()
    return args

def main():
    # 遺伝的アルゴリズムのパラメータ
    random.seed(0)
    args = opts()

    # 初期集団を生成
    population = toolbox.population(n=args.pop)

    # 適応度を計算
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)
    
    # 世代数分進化を繰り返す
    for gen in range(args.ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=args.cxpb, mutpb=args.mutpb) # 交叉と突然変異
        for child in offspring:
            child.fitness.values = toolbox.evaluate(child) # 適応度の計算

        population = toolbox.select(population + offspring, len(population)) # 次世代の選択

        # 適応度の統計情報を出力
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print(f'gen: {gen}, avg: {mean}, std: {std}')
    
    # 最終世代の適応度を出力
    top10 = tools.selBest(population, 10)
    for i, ind in enumerate(top10):
        seq = ''.join(ind)
        print(f'rank: {i}, {seq}, {ind.fitness.values}')


if __name__ == '__main__':
    main()
    print('Done')