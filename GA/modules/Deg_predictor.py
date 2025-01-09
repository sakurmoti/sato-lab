'''
Copyright (c) 2024 Shujun-He
Released under the MIT license
https://github.com/Shujun-He/RibonanzaNet/blob/main/LICENSE

This code is based on the following code:
https://www.kaggle.com/code/shujun717/ribonanzanet-deg
'''

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os,sys,pathlib,yaml
from .NN import Network, dropout

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(Network.RibonanzaNet):
    def __init__(self, config):
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.decoder = nn.Linear(config.ninp,5)
        
    def forward(self,src):
        
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        output=self.decoder(sequence_features) #predict

        return output#.squeeze(-1)
    

class predictor():
    def __init__(self):
        configPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'pairwise.yaml')
        modelPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'RibonanzaNet-Deg.pt')
        # configPath = pathlib.Path(__file__).parent / 'pairwise.yaml'
        # modelPath = pathlib.Path(__file__).parent / 'RibonanzaNet-Deg.pt'
        self.model = finetuned_RibonanzaNet(load_config_from_yaml(configPath)).cuda()
        self.model.load_state_dict(torch.load(modelPath, map_location='cuda', weights_only=True))
        self.model.eval()
        # self.tokens = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.tokens = [0]*128
        self.tokens[ord('A')] = 0
        self.tokens[ord('C')] = 1
        self.tokens[ord('G')] = 2
        self.tokens[ord('T')] = 3


    def predict(self, sequence):
        example = np.array([self.tokens[ord(nt)] for nt in sequence])
        # print(example)
        seq = torch.tensor(example, device='cuda').unsqueeze(0)
        # print(seq)
        with torch.no_grad():
            return self.model(seq).cpu().numpy()
        
def score_sum(score, label='deg_50C'):
    sum = 0
    if(label == 'reactivity'):
        for i in range(len(score[0])):
            sum += score[0][i][0]
        return sum
    elif(label == 'deg_Mg_pH10'):
        for i in range(len(score[0])):
            sum += score[0][i][1]
        return sum
    elif(label == 'deg_pH10'):
        for i in range(len(score[0])):
            sum += score[0][i][2]
        return sum
    elif(label == 'deg_Mg_50C'):
        for i in range(len(score[0])):
            sum += score[0][i][3]
        return sum
    elif(label == 'deg_50C'):
        for i in range(len(score[0])):
            sum += score[0][i][4]
        return sum
    else:
        raise ValueError(f'Invalid label, {label}')

def score_to_df(score):
    labels = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    # predictの結果scoreが(1, length, 5)のshapeで返ってくる
    # lengthはsequenceの長さ、5はlabelsの数
    if score.shape[0] != 1:
        raise ValueError(f'Invalid shape, {score.shape}')

    df = pd.DataFrame(score[0], columns=labels)
    return df