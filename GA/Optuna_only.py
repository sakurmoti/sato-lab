import optuna
import random

# 配列をランダムに生成する関数
def generate_random_sequence():
    return "".join(random.choice("ATCG") for _ in range(50))

# RNA配列に対する目的関数 (f(seq), g(seq))
def f(seq):
    # ユーザーが定義した目的関数1
    return sum(1 for c in seq if c == "A")  # 例: Aの数を最大化

def g(seq):
    # ユーザーが定義した目的関数2
    return sum(1 for c in seq if c == "G")  # 例: Gの数を最大化

# Optunaの目的関数
def objective(trial):
    # 配列長を固定（例: 50）し、各塩基をサンプリング
    sequence = "".join(trial.suggest_categorical(f"base_{i}", ["A", "T", "C", "G"]) for i in range(50))
    
    # 2つの目的関数を計算
    obj1 = f(sequence)
    obj2 = g(sequence)
    
    # 複数目的関数のリストを返す
    return obj1, obj2

# Optunaの多目的最適化用スタディを設定
study = optuna.create_study(directions=["maximize", "maximize"], sampler=optuna.samplers.NSGAIIISampler())

# 最適化の実行
study.optimize(objective, n_trials=1000)

# 最適な解（Paretoフロント）の出力
print("Pareto front solutions:")
for trial in study.best_trials:
    print(f"Values: {trial.values}, Sequence: {''.join(trial.params.values())}")
