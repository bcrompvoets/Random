import time
tic = time.perf_counter()
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from custom_dataloader import replicate_data_single
from NN_Defs import BaseMLP, TwoLayerMLP, FiveLayerMLP, find_best_MLP, MLP_data_setup, preproc_yso
import multiprocessing as mp

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')

# YSO_EG_Stars Train
X_tr = np.load("../Data/c2d_1k_INP.npy") # Load input data
Y_tr = np.load("../Data/c2d_1k_TAR.npy") # Load target data
X_tr = np.float32(X_tr)
Y_tr = np.float32(Y_tr)
Y_tr = preproc_yso(alph=X_tr[:,-1],tar=Y_tr)
inp_tr, tar_tr = replicate_data_single(X_tr, Y_tr, [len(np.where(Y_tr==0.)[0])]*len(np.unique(Y_tr)))

# YSO_EG_Stars Valid
X_va = np.load("../Data/NGC2264_INP.npy") # Load input data
Y_va = np.load("../Data/NGC2264_TAR.npy") # Load target data
X_va = np.float32(X_va)
Y_va = np.float32(Y_va)
Y_va = preproc_yso(alph=X_va[:,-1],tar=Y_va)
inp_va, tar_va = replicate_data_single(X_va, Y_va,\
     [len(np.where(Y_va==0.)[0]),len(np.where(Y_va==1.)[0]),len(np.where(Y_va==2.)[0]),len(np.where(Y_va==3.)[0]),len(np.where(Y_va==4.)[0]),len(np.where(Y_va==5.)[0])])



# YSO_EG_Stars Test
X_te = np.load("../Data/c2d_10k_INP.npy") # Load input data
Y_te = np.load("../Data/c2d_10k_TAR.npy") # Load target data
X_te = np.float32(X_te)
Y_te = np.float32(Y_te)
Y_te = preproc_yso(alph=X_te[:,-1],tar=Y_te)
inp_te, tar_te = replicate_data_single(X_te, Y_te,\
      [len(np.where(Y_te==0.)[0]),len(np.where(Y_te==1.)[0]),len(np.where(Y_te==2.)[0]),len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])

inp_tr = np.delete(inp_tr,np.s_[8:10],axis=1)
inp_va = np.delete(inp_va,np.s_[8:10],axis=1)
inp_te = np.delete(inp_te,np.s_[8:10],axis=1)
# inp_tr = inp_tr[np.where(inp_tr[:,9]!=-99)[0]]
# tar_tr = tar_tr[np.where(inp_tr[:,9]!=-99)[0]]
# inp_va = inp_va[np.where(inp_va[:,9]!=-99)[0]]
# tar_va = tar_va[np.where(inp_va[:,9]!=-99)[0]]
# inp_te = inp_te[np.where(inp_te[:,9]!=-99)[0]]
# tar_te = tar_te[np.where(inp_te[:,9]!=-99)[0]]

train_loader, val_loader, test_loader = MLP_data_setup(inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te)

custom_labs = ['YSO - Class I','YSO - Class FS','YSO - Class II','YSO - Class III','EG','Star']

if __name__ == '__main__':
    momentum_vals = np.array([0.6, 0.75, 0.9])
    learning_rate_vals = np.array([1e-1, 1e-2, 1e-3])
    epochs = 3000
    columns = 9
    filepath = "../Results/MLP/YSO_1k_CIII/"
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