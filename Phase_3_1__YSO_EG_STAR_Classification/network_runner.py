import time
tic = time.perf_counter()
# Check to make sure running on M1, will say 'arm'
import platform
print(platform.processor())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms


import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, recall_score, f1_score

from custom_dataloader import replicate_data
from NN_Defs import get_n_params, train, validate, BaseMLP, TwoLayerMLP, FiveLayerMLP, find_best_MLP
import multiprocessing as mp

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')

# settings for plotting in this section
cm_blues = plt.cm.Blues
custom_labs = ['YSO','EG','Star']

# data load
X = np.load("Data_and_Results/Inputs_YSO_EG_Stars.npy") # Load input data
X = np.float32(X)
Y = np.load("Data_and_Results/Targets_YSO_EG_Stars.npy") # Load target data
Y = np.float32(Y)

seed_val = 1111
train_amount = [1472,857,1257]
valid_amount = [613,405,4359]

inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te = replicate_data(X, Y, train_amount, valid_amount)

# scaling data according to training inputs
scaler_S = StandardScaler().fit(inp_tr)
inp_tr = scaler_S.transform(inp_tr)
inp_va = scaler_S.transform(inp_va)
inp_te = scaler_S.transform(inp_te)

# concatenate the labels onto the inputs for both training and validation
inp_tr = torch.as_tensor(inp_tr)
tar_tr = torch.as_tensor(tar_tr)
inp_va = torch.as_tensor(inp_va)
tar_va = torch.as_tensor(tar_va)
inp_te = torch.as_tensor(inp_te)
tar_te = torch.as_tensor(tar_te)

train_data = data_utils.TensorDataset(inp_tr, tar_tr)
val_data = data_utils.TensorDataset(inp_va, tar_va)
test_data = data_utils.TensorDataset(inp_te, tar_te)

# constructing data loaders for nn
train_loader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True,pin_memory=True,num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=25, shuffle=True,pin_memory=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=25, shuffle=True,pin_memory=True,num_workers=0)


if __name__ == '__main__':

    momentum_vals = [0.6,0.75, 0.9]
    learning_rate_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_size = [25,100,200]
    epochs = 3000
    filepath = "MLP_Runs_Results/UnEvenSplit/"
    filepaths = [filepath+"OneLayer/", filepath+"TwoLayer/", filepath+"FiveLayer/"]

    # We want to run a loop over the momentum and learning rate values, and use the
    # validation f1 score for YSOs as the metric at which to determine the best 


    iters = [(BaseMLP, filepaths[0], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, device),\
        (TwoLayerMLP, filepaths[1], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, device),\
        (FiveLayerMLP,filepaths[2],learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, device)]
    with mp.Pool(4) as pool:
        bestfiles = pool.starmap(find_best_MLP, iters)

    print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")