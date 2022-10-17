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
# print(f"YSO shape: {input[input['Target']==0].shape}")
# print(f"EG shape: {input[input['Target']==1].shape}")
# print(f"Star shape: {input[input['Target']==2].shape}")


# input['Target']=preproc_yso(input[['alpha']].values,input[['Target']].values,CIII)

inp_tr, tar_tr,inp_va, tar_va,inp_te, tar_te = replicate_data(input[bands].values.astype(float), input[['Target']].values.astype(int),[3000]*3,[1000,1650,10000])# 171,133,693,219,1974,2500
while np.all(np.isfinite(inp_tr)) == False:
    inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te = replicate_data(input[bands].values.astype(float), input[['Target']].values.astype(int),[3000]*3,[1000,1650,10000])# 171,133,693,219,1974,2500
# print('Assigned training, validation, and test sets.')



# custom_labs = ['YSO - Class I','YSO - Class FS','YSO - Class II','YSO - Class III','EG','Star']

# custom_labs = ['YSO - Class I','YSO - Class FS','YSO - Class II','YSO - Class III']

custom_labs = ['YSO','EG','Star']

if __name__ == '__main__':
    momentum_vals = np.array([0.6, 0.75, 0.9])
    learning_rate_vals = np.array([1e-3])#1e-1, 1e-2, 
    epochs = 2500
    filepath = "../Results/c2d_quality_17_LR_Reduce_YSE/"
    # filepaths = [filepath+"OneLayer_", filepath+"TwoLayer_", filepath+"FiveLayer_"]



    filepaths = [filepath+"IRAC_alpha", filepath+"IRAC_MIPS_alpha", filepath+ "IRAC_2MASS_alpha", filepath+"IRAC_MIPS_2MASS_alpha"]

    band = [[idx for idx in input.columns.values if (idx[-2].lower() == 'R'.lower() or idx=='alpha')],\
        [idx for idx in input.columns.values if (idx[-2].lower() == 'R'.lower() or idx[-2].lower() == 'P'.lower() or idx=='alpha')],\
        [idx for idx in input.columns.values if (idx[-2].lower() == 'R'.lower() or idx[-2].lower() != 'P'.lower() or idx=='alpha')],\
        bands] # IRAC only, IRAC and MIPS, IRAC and 2MASS, 

    train_loader = val_loader = test_loader = [0]*4

    iters = [0]*4




    for i in [0,1,2,3]:
        band_ind = np.array(np.where(np.isin(bands,band[i]))[0])
        columns = len(band_ind)
        train_loader, val_loader, test_loader = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_va[:,band_ind], tar_va, inp_va[:,band_ind], tar_va)
        iters[i] = (TwoLayerMLP, filepaths[i], epochs, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device)


    # We want to run a loop over the momentum and learning rate values, and use the
    # validation f1 score for YSOs as the metric at which to determine the best 
    # iters = [(TwoLayerMLP, filepaths[1], epochs, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device)]#,\
        # (BaseMLP, filepaths[0], epochs, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device),\
        # (FiveLayerMLP,filepaths[2], epochs, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, columns, custom_labs, device)]
    with mp.Pool(4) as pool:
        bestfiles = pool.starmap(find_best_MLP, iters)

    print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")