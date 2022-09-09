import time
tic = time.perf_counter()
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from custom_dataloader import replicate_data
from NN_Defs import BaseMLP, TwoLayerMLP, FiveLayerMLP, find_best_MLP, MLP_data_setup, preproc_yso
import multiprocessing as mp

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')
CIII = True # Train with Class III as a possible class
# YSO_EG_Stars Train
file_inp = "c2d_w_quality.csv"
input = pd.read_csv(file_inp)
bands = [idx for idx in input.columns.values if (idx[0].lower() == 'm'.lower() or idx[0].lower() == 'e'.lower())]
bands = bands[:-2]
bands.append("alpha")
band = bands#[idx for idx in input.columns.values if idx[-2].lower() != 'P'.lower()]
band_ind = np.array(np.where(np.isin(bands,band))[0])
bands = np.array(bands)
print(bands[band_ind])
# print(f"YSO shape: {input[input['Target']==0].shape}")
# print(f"EG shape: {input[input['Target']==1].shape}")
# print(f"Star shape: {input[input['Target']==2].shape}")
inp_tr, tar_tr,inp_va, tar_va,inp_te, tar_te = replicate_data(input[bands[band_ind]].values.astype(float), input[['Target']].values.astype(int),[3000]*3,[1000,1650,10000])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])
while np.all(np.isfinite(inp_tr)) == False:
    inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te = replicate_data(input[bands[band_ind]].values.astype(float), input[['Target']].values.astype(int),[3000]*3,[1000,1650,10000])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])
# print('Assigned training, validation, and test sets.')

# FOR YSOs ONLY
preproc_yso(inp_tr[:,-1],tar_tr,CIII)
preproc_yso(inp_va[:,-1],tar_va,CIII)
preproc_yso(inp_te[:,-1],tar_te,CIII)
ys_ind_tr = np.where(tar_tr<=3)[0]
ys_ind_va = np.where(tar_va<=3)[0]
ys_ind_te = np.where(tar_te<=3)[0]
inp_tr = inp_tr[ys_ind_tr,:]
tar_tr = tar_tr[ys_ind_tr]
inp_va = inp_va[ys_ind_va,:]
tar_va = tar_va[ys_ind_va]
inp_te = inp_te[ys_ind_te,:]
tar_te = tar_te[ys_ind_te]


train_loader, val_loader, test_loader = MLP_data_setup(inp_tr, tar_tr, inp_va, tar_va, inp_va, tar_va)

# custom_labs = ['YSO - Class I','YSO - Class FS','YSO - Class II','YSO - Class III','EG','Star']

custom_labs = ['YSO - Class I','YSO - Class FS','YSO - Class II','YSO - Class III']

# custom_labs = ['YSO','EG','Star']

if __name__ == '__main__':
    momentum_vals = np.array([0.6, 0.75, 0.9])
    learning_rate_vals = np.array([1e-1, 1e-2, 1e-3])
    epochs = 2500
    columns = inp_tr.shape[1]
    filepath = "Results/c2d_quality_11_IRAC_MIPS_2MASS_alpha_YSO/"
    filepaths = [filepath+"OneLayer_", filepath+"TwoLayer_", filepath+"FiveLayer_"]

    # We want to run a loop over the momentum and learning rate values, and use the
    # validation f1 score for YSOs as the metric at which to determine the best 
    iters = [(BaseMLP, filepaths[0], epochs, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device),\
        (TwoLayerMLP, filepaths[1], epochs, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device)]#,\
        # (FiveLayerMLP,filepaths[2], epochs, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device)]
    with mp.Pool(2) as pool:
        bestfiles = pool.starmap(find_best_MLP, iters)

    print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")