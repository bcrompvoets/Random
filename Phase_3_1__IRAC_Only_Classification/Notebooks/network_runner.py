import time

tic = time.perf_counter()
# Check to make sure running on M1, will say 'arm'
import platform
print(platform.processor())

import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from custom_dataloader import replicate_data, replicate_data_single
from NN_Defs import BaseMLP, TwoLayerMLP, FiveLayerMLP, find_best_MLP, MLP_data_setup,main
import multiprocessing as mp

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')

# YSO_EG_Stars Train
# X_tr = np.load("../Data_and_Results/c2d_Inputs_CLOUDS_Train.npy") # Load input data
# Y_tr = np.load("../Data_and_Results/c2d_Targets_CLOUDS_Train.npy") # Load target data
# inp_tr, tar_tr = replicate_data_single(X_tr, Y_tr, [1082,1082,1082])

X_tr = np.load("../Data_and_Results/Inputs_YSO_EG_Stars_alpha.npy") # Load input data
Y_tr = np.load("../Data_and_Results/Targets_YSO_EG_Stars_alpha.npy") # Load target data
X_tr = np.float32(X_tr)
Y_tr = np.float32(Y_tr)
inp_tr, tar_tr = replicate_data_single(X_tr, Y_tr, [1500,1500,1500])

# YSO_EG_Stars Valid
X_va = np.load("../Data_and_Results/c2d_Inputs_CORES_Valid.npy") # Load input data
Y_va = np.load("../Data_and_Results/c2d_Targets_CORES_Valid.npy") # Load target data
X_va = np.float32(X_va)
Y_va = np.float32(Y_va)
inp_va, tar_va = replicate_data_single(X_va, Y_va, [314,212,4269])

# YSO_EG_Stars Test
inp_te = np.load("../Data_and_Results/c2d_Inputs_CLOUDS_Train.npy")#Rap_Inputs_Test.npy") # Load input data
tar_te = np.load("../Data_and_Results/c2d_Targets_CLOUDS_Train.npy")#Rap_Targets_Test.npy") # Load target data
inp_te = np.float32(inp_te)
tar_te = np.float32(tar_te)
# inp_te, tar_te = replicate_data_single(X_te, Y_te, [553,373,7518])

# YSO 
# X = np.load("../Data_and_Results/Inputs_YSO_Train.npy") # Load input data
# Y = np.load("../Data_and_Results/Targets_YSO_Train.npy") # Load target data
# X = np.float32(X)
# Y = np.float32(Y)
# YSO only vals
# train_amount = [3000,3000,3000,3000]
# valid_amount = [6726,25687,10162,2300]


train_loader, val_loader, test_loader = MLP_data_setup(inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te)

custom_labs = ['YSO','EG','Star']
# custom_labs = ['Class I', 'Class II', 'Flat-Spectrum', 'Class III']

if __name__ == '__main__':
    momentum_vals = [0.6, 0.75, 0.9]
    learning_rate_vals = [1e-1, 1e-2, 1e-3]
    epochs = 3000
    filepath = "../MLP_Runs_Results/YSO_EG_Stars/c2dValid/"
    filepaths = [filepath+"OneLayer_", filepath+"TwoLayer_", filepath+"FiveLayer_"]

    # We want to run a loop over the momentum and learning rate values, and use the
    # validation f1 score for YSOs as the metric at which to determine the best 
    iters = [(BaseMLP, filepaths[0], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, custom_labs, device),\
        (TwoLayerMLP, filepaths[1], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, custom_labs, device),\
        (FiveLayerMLP,filepaths[2],learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, custom_labs, device)]
    with mp.Pool(6) as pool:
        bestfiles = pool.starmap(find_best_MLP, iters)

    print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")