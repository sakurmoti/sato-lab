'''
Copyright (c) 2024 Shujun-He
Released under the MIT license
https://github.com/Shujun-He/RibonanzaNet/blob/main/LICENSE
'''

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import yaml

sys.path.append('./NN')
from Network import *

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

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.decoder = nn.Linear(config.ninp,5)
        
    def forward(self,src):
        
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        output=self.decoder(sequence_features) #predict

        return output#.squeeze(-1)
    

class RNA2D_predictor():
    def __init__(self):
        self.model = finetuned_RibonanzaNet(load_config_from_yaml("./pairwise.yaml")).cuda()
        self.model.load_state_dict(torch.load('./RibonanzaNet-Deg.pt', map_location='cpu'))
        
    def predict(self, sequence):
        self.model.eval()

        tokens = {nt:i for i,nt in enumerate('ACGU')}
        example = [tokens[nt] for nt in sequence]
        example = np.array(example)
        example = torch.tensor(example)

        seq = example.cuda().unsqueeze(0)
        with torch.no_grad():
            return self.model(seq).cpu().numpy()