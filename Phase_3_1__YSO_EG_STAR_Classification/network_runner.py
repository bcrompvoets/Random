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
from NN_Defs import get_n_params, train, validate, BaseMLP, TwoLayerMLP, FiveLayerMLP


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

def main(epochs, NetInstance, OptInstance, outfile, ScheduleInstance=None):

    train_loss_all = []
    val_loss_all = []
    
    for epoch in range(0, epochs):
        train_loss, train_predictions, train_truth_values = train(epoch, NetInstance, OptInstance, train_loader, device)
        val_loss, val_predictions, val_truth_values = validate(NetInstance, val_loader, device)
        
        # store loss in an array to plot
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)

        if ScheduleInstance is not None:
            ScheduleInstance.step()

        # print outs
        if epoch % 1000 == 0:
            print(f'Train Epoch: {epoch} ----- Train Loss: {train_loss.item():.6f}')
            print(f'Validation Loss: {val_loss:.4f}')

            if ScheduleInstance is not None:
                print(f'Learning Rate : {ScheduleInstance.get_last_lr()}')
    
    # running testing set through network
    test_loss, test_predictions, test_truth_values = validate(NetInstance, test_loader, device)

    # plotting losses and saving fig
    # fig, ax = plt.subplots(figsize=(10,6))
    # ax.plot(train_loss_all, label='Train Loss')
    # ax.plot(val_loss_all, label='Validation Loss')
    # ax.set_xlabel('Epoch')
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # plt.savefig(outfile+'_loss.png')
    # plt.close()
    
    # plotting Confusion Matrix and saving
    # fig, ax = plt.subplots(3,1, figsize=(12,20))
    # ConfusionMatrixDisplay.from_predictions(train_truth_values, train_predictions, ax = ax[0], normalize='true', cmap=cm_blues, display_labels=custom_labs, colorbar=False)
    # ConfusionMatrixDisplay.from_predictions(val_truth_values, val_predictions, ax = ax[1], normalize='true', cmap=cm_blues, display_labels=custom_labs, colorbar=False)
    # ConfusionMatrixDisplay.from_predictions(test_truth_values, test_predictions, ax = ax[2], normalize='true', cmap=cm_blues, display_labels=custom_labs, colorbar=False)
    
    # setting plotting attributes here
    # ax[0].set_title('Training Set')
    # ax[1].set_title('Validation Set')
    # ax[2].set_title('Testing Set')
    # plt.tight_layout()
    # plt.savefig(outfile+'_CM.png')
    # plt.close()
    
    # printout for classifcation report for all sets
    # print('Target Set Report')
    # print(classification_report(train_truth_values, train_predictions, target_names=custom_labs, zero_division=0))
    print('Validation Set Report')
    print(classification_report(val_truth_values, val_predictions, target_names=custom_labs, zero_division=0))
    # print('Testing Set Report')
    # print(classification_report(test_truth_values, test_predictions, target_names=custom_labs, zero_division=0))

    # saving the network settings
    torch.save(NetInstance.state_dict(),outfile+'_Settings')

    return f1_score(val_truth_values,val_predictions,average=None,zero_division=1)

if __name__ == '__main__':

    momentum_vals = [0.6,0.75, 0.9]
    learning_rate_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_size = [25,100,200]
    epochs = 3000
    filepath = "MLP_Runs_Results/UnEvenSplit/"
    filepaths = [filepath+"OneLayer/", filepath+"TwoLayer/", filepath+"FiveLayer/"]

    # We want to run a loop over the momentum and learning rate values, and use the
    # validation f1 score for YSOs as the metric at which to determine the best 
    def find_best_MLP(MLP, filepath_to_MLPdir, learning_rate_vals, momentum_vals)
        f1Max = 0.5
        print(filepath_to_MLPdir)
        for l, lr in enumerate(learning_rate_vals):
            for m, mo in enumerate(momentum_vals):
                for n in [10,20]:
                    tic1 = time.perf_counter()
                    outfile = filepath_to_MLPdir + "LR_" + str(lr) + "_MO_" + str(mo)
                    NN = MLP(8,n,3)
                    optimizer = optim.SGD(NN.parameters(), lr=lr, momentum=mo)
                    f1score = main(3000, NN, optimizer, outfile)
                    if f1score[0] > f1Max and f1score[1] != 0 and f1score[2] != 0:
                        f1Max = f1score[0]
                        bestfile = outfile
                        print(outfile)
                    toc1 = time.perf_counter()
                    print(f"Completed in {(toc1 - tic1)/60:0.1f} minutes")
        return bestfile

    iters = [(BaseMLP, filepaths[0], learning_rate_vals, momentum_vals), (TwoLayerMLP, filepaths[1], learning_rate_vals, momentum_vals),\
        (FiveLayerMLP,filepaths[i],learning_rate_vals, momentum_vals)]
    with mp.Pool(4) as pool:
        bestfiles = pool.starmap(find_best_MLP, iters)

    print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")