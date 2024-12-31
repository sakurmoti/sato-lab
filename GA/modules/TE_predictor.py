import torch
import torch.nn as nn
import numpy as np
import random

import esm
from esm.data import *
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

class CNN_linear(nn.Module):
    def __init__(self, alphabet, 
                 border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):
        
        super(CNN_linear, self).__init__()
        
        self.embedding_size = 128
        self.layers = 6
        self.heads = 16
        self.batch_toks = 4096

        self.border_mode = border_mode
        self.inp_len = 100
        self.nodes = 40
        self.cnn_layers = 0
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = 0.2
        
        self.esm2 = ESM2(num_layers = self.layers,
                                embed_dim = self.embedding_size,
                                attention_heads = self.heads,
                                alphabet = alphabet)
    
        self.conv1 = nn.Conv1d(in_channels = self.embedding_size, 
                      out_channels = self.nbr_filters, kernel_size = self.filter_len, padding = self.border_mode)
        self.conv2 = nn.Conv1d(in_channels = self.nbr_filters, 
                      out_channels = self.nbr_filters, kernel_size = self.filter_len, padding = self.border_mode)
        
        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(in_features = self.embedding_size, out_features = self.nodes)
        self.linear = nn.Linear(in_features = self.nbr_filters, out_features = self.nodes)
        
        self.output = nn.Linear(in_features = self.nodes, out_features = 1)
        if self.cnn_layers == -1: self.direct_output = nn.Linear(in_features = self.embedding_size, out_features = 1)


    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation = True):
        
        x = self.esm2(tokens, [self.layers], need_head_weights, return_contacts, return_representation)
        x = x["representations"][self.layers][:, 0]
        x_o = x.unsqueeze(2)
    
        if self.cnn_layers >= 1:
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2: 
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3: 
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)
        
#         if self.cnn_layers >= 1: 
        x = self.flatten(x_o)
        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x)
            else:
                o_linear = self.fc(x)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x)

        # print(f'output shape: {o.shape}')
        return o 

class predictor():
    def __init__(self):
        self.alphabet = Alphabet(mask_prob=0.0, standard_toks='AGCT')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = CNN_linear(self.alphabet).to(self.device)
        modelpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'TEmodel.pt')
        model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(modelpath, weights_only=True).items()})
        model.eval()
        self.model = model

    def predict(self, seq):
        with torch.no_grad():
            toks = self.alphabet.encode(seq)
            toks = torch.tensor(toks).to(self.device)
            toks = toks.unsqueeze(0)
            outputs = self.model(toks, return_representation = True, return_contacts = True) # scaled_log2
            y_pred = outputs.reshape(-1).cpu().detach().tolist()
            return y_pred[0]
