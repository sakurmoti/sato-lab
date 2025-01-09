import optuna
from modules import Deg_predictor, TE_predictor

Degpred = Deg_predictor.predictor()
TEpred = TE_predictor.predictor()

# Optunaの目的関数
def objective(trial):
    # 配列長を固定（例: 50）し、各塩基をサンプリング
    length = 50
    sequence = ''.join([trial.suggest_categorical(f"base{i}", ['A', 'T', 'C', 'G']) for i in range(length)])
    
    # 2つの目的関数を計算
    # te = TEpred.predict(sequence)
    # deg = Deg_predictor.score_sum(Degpred.predict(sequence))

    te = (sequence.count('A'))
    deg = (sequence.count('C'))
    # 複数目的関数のリストを返す
    return te, deg

# Optunaの多目的最適化用スタディを設定
study = optuna.create_study(
    directions=["maximize", "minimize"],
)

# 最適化の実行
study.optimize(objective, n_trials=100)

savepath = "/home/sato-lab.org/takayu/project/sato-lab/slurm/logs/"
fig1 = optuna.visualization.plot_pareto_front(study, target_names=["A", "C"])
fig1.write_image(savepath + "img/pareto_front.png")

fig2 = optuna.visualization.plot_param_importances(study)
fig2.write_image(savepath + "img/param_importances.png")

print(f'Number of trials on the Pareto front: {len(study.best_trials)}')
for i, trial in enumerate(study.best_trials):
    print(f'Trial {i}:')
    print(f'  - TE: {trial.values[0]}')
    print(f'  - Deg: {trial.values[1]}')
    seq = ''.join([trial.params[f'base{i}'] for i in range(50)])
    print(f'  - Sequence: {seq}')