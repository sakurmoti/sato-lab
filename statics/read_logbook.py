import pickle

path = '../slurm/logs/deap/img/logbook715_TEonly.pkl'
with open(path, 'rb') as f:
    logbook = pickle.load(f)
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_maxs = logbook.select("max")
    fit_avgs = logbook.select("avg")

for i in range(len(gen)):
    print(f"gen: {gen[i]}, min: {fit_mins[i][0]}, max: {fit_maxs[i][0]}, avg: {fit_avgs[i][0]}")
    