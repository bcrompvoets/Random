import time
tic = time.perf_counter()
# Check to make sure running on M1, will say 'arm'
import platform
print(platform.processor())

import torch
import numpy as np
from custom_dataloader import replicate_data, replicate_data_single
from NN_Defs import BaseMLP, TwoLayerMLP, FiveLayerMLP, find_best_MLP, MLP_data_setup
import multiprocessing as mp
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')

# YSO_EG_Stars Train
X_tr = np.load("../Data_and_Results/c2d_ALL_HT_Inputs.npy") # Load input data
Y_tr = np.load("../Data_and_Results/c2d_ALL_HT_Targets.npy") # Load target data
X_tr = np.float32(X_tr)
Y_tr = np.float32(Y_tr)
inp_tr, tar_tr = replicate_data_single(X_tr, Y_tr, [10000,10000,10000])

# YSO_EG_Stars Valid
X_va = np.load("../Data_and_Results/c2d_Inputs_CORES_Valid.npy")[:,[0,2,4,6]] # Load input data
Y_va = np.load("../Data_and_Results/c2d_Targets_CORES_Valid.npy") # Load target data
inp_va = np.float32(X_va)
tar_va = np.float32(Y_va)
# inp_va, tar_va = replicate_data_single(X_va, Y_va, [314,212,4269])

# YSO_EG_Stars Test
X_te = np.load("../Data_and_Results/Rap_Inputs_Test.npy")[:,[0,2,4,6]] # Load input data
Y_te = np.load("../Data_and_Results/Rap_Targets_Test.npy") # Load target data
inp_te = np.float32(X_te)
tar_te = np.float32(Y_te)
# inp_te, tar_te = replicate_data_single(X_te, Y_te, [553,373,7518]) # We still use this as it shuffles the data


# # YSO 
# X = np.load("../Data_and_Results/Inputs_YSO_Train.npy") # Load input data
# Y = np.load("../Data_and_Results/Targets_YSO_Train.npy") # Load target data
# X = np.float32(X)
# Y = np.float32(Y)
# # YSO only vals
# train_amount = [3000,3000,3000,3000]
# valid_amount = [6726,25687,10162,2300]
# inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te = replicate_data(X,Y,train_amount,valid_amount)


train_loader, val_loader, test_loader = MLP_data_setup(inp_tr, tar_tr, inp_te, tar_te, inp_va, tar_va)

custom_labs = ['YSO','EG','Star']
# custom_labs = ['Class I', 'Class II', 'Flat-Spectrum', 'Class III']

if __name__ == '__main__':
    momentum_vals = np.array([0.6, 0.75, 0.9])
    learning_rate_vals = np.array([1e-1, 1e-2, 1e-3])
    epochs = 3000
    columns = 4
    filepath = "../MLP_Runs_Results/YSO_EG_Stars/c2d_ALL_HT_10000/"
    filepaths = [filepath+"OneLayer_", filepath+"TwoLayer_", filepath+"FiveLayer_"]

    # We want to run a loop over the momentum and learning rate values, and use the
    # validation f1 score for YSOs as the metric at which to determine the best 
    iters = [(BaseMLP, filepaths[0], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device),\
        (TwoLayerMLP, filepaths[1], learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device),\
        (FiveLayerMLP,filepaths[2],learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device)]
    with mp.Pool(3) as pool:
        bestfiles = pool.starmap(find_best_MLP, iters)

    print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")