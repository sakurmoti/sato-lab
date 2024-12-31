import sys
sys.path.append('../')
from GA.modules import TE_predictor

import pandas as pd
data = pd.read_csv('../GA/modules/data/mixed_sequence.csv')
df = data.__deepcopy__()

print(df.head())

predictor = TE_predictor.predictor()

import matplotlib.pyplot as plt

# Predict using the predictor
df['predicted_te_log'] = df['utr'].apply(lambda seq: predictor.predict(seq))

print(df.head())
print(df['te_log'].describe())
print(df['predicted_te_log'].describe())

# Plot the distribution of predicted and true values
plt.figure(figsize=(10, 6))
plt.hist(df['te_log'], bins=50, alpha=0.5, label='True TE Log')
plt.hist(df['predicted_te_log'], bins=50, alpha=0.5, label='Predicted TE Log')
plt.xlabel('TE Log')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Distribution of True and Predicted TE Log Values')
plt.savefig('../slurm/logs/test/te_pred.png')