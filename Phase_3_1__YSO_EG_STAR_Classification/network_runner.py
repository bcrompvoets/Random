import time

from zmq import int_sockopt_names

tic = time.perf_counter()
# Check to make sure running on M1, will say 'arm'
import platform
print(platform.processor())

import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from custom_dataloader import replicate_data
from NN_Defs import BaseMLP, TwoLayerMLP, FiveLayerMLP, find_best_MLP, MLP_data_setup
import multiprocessing as mp

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')

# YSO_EG_Stars
# X = np.load("Data_and_Results/Inputs_YSO_EG_Stars.npy") # Load input data
# Y = np.load("Data_and_Results/Targets_YSO_EG_Stars.npy") # Load target data

# YSO 
X = np.load("Data_and_Results/Inputs_YSO_Train.npy") # Load input data
Y = np.load("Data_and_Results/Targets_YSO_Train.npy") # Load target data


X = np.float32(X)
Y = np.float32(Y)

# YSO only vals
train_amount = [3000,3000,3000,3000]
valid_amount = [2271,440,3362]

inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te = replicate_data(X, Y, train_amount, valid_amount)

# Test
inp_te = np.load("Data_and_Results/Inputs_YSO_Test.npy") # Load input data
tar_te = np.load("Data_and_Results/Targets_YSO_Test.npy") # Load target data

# scaling data according to training inputs
scaler_S = StandardScaler().fit(inp_tr)
inp_tr = scaler_S.transform(inp_tr)
inp_te = scaler_S.transform(inp_te) 

inp_te = torch.as_tensor(inp_te)
tar_te = torch.as_tensor(tar_te)

# pass tensors into TensorDataset instances
test_data = data_utils.TensorDataset(inp_te, tar_te)

# constructing data loaders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=25, shuffle=True)


if __name__ == '__main__':

    train_loader, val_loader, test_loader = MLP_data_setup(X, Y, train_amount,valid_amount)

    momentum_vals = [0.6,0.75, 0.9]
    learning_rate_vals = [1e-1, 1e-2, 1e-3, 1e-4]
    epochs = 3000
    filepath = "MLP_Runs_Results/YSO/"
    filepaths = [filepath+"OneLayer/", filepath+"TwoLayer/", filepath+"FiveLayer/"]

    # We want to run a loop over the momentum and learning rate values, and use the
    # validation f1 score for YSOs as the metric at which to determine the best 
    iters = [(BaseMLP, filepaths[0], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, device),\
        (TwoLayerMLP, filepaths[1], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, device),\
        (FiveLayerMLP,filepaths[2],learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, device)]
    with mp.Pool(6) as pool:
        bestfiles = pool.starmap(find_best_MLP, iters)

    print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")