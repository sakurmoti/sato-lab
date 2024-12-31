import sys
sys.path.append('../')
from GA.modules import Deg_predictor
import pandas as pd

test_data = pd.read_json('../GA/modules/data/test.json', lines=True)
test_dataset = test_data['sequence'].values

import tqdm

pred = Deg_predictor.predictor()
test_preds = []
for i in range(10):
    test_preds.append(pred.predict(test_dataset[i]))

print(test_preds[0].shape)

import matplotlib.pyplot as plt
lables = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

for i in range(5):
    plt.plot(test_preds[0][0,:,i], label=lables[i])

plt.legend()
plt.savefig('../slurm/logs/test/pred_test.png')

preds = []
ids = []
for i in range(10):
    preds.append(test_preds[i][0,:])
    id = test_data.loc[i, 'id']
    ids.extend([f"{id}_{pos}" for pos in range(len(test_preds[i][0,:]))])

import numpy as np
preds = np.concatenate(preds)
print(preds.shape)

sub = pd.DataFrame()
sub['id_seqpos'] = ids
for i,l in enumerate(lables):
    sub[l] = preds[:,i]

sub.to_csv('../slurm/logs/test/submission_test.csv', index=False)