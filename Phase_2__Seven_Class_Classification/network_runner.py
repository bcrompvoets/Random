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
from sklearn.metrics import classification_report

from custom_dataloader import replicate_data
from NN_Defs import get_n_params, train, validate, BaseMLP, TwoLayerMLP


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')

# settings for plotting in this section
cm_blues = plt.cm.Blues
custom_labs = ['Class 1', 'Class 2', 'Gals','AGNs','Shocks','PAHs','Stars']

# data load
X = np.load("Input_Class_AllClasses_Sep.npy") # Load input data
X = np.float32(X)
Y = np.load("Target_Class_AllClasses_Sep.npy") # Load target data
Y = np.float32(Y)

seed_val = 1111
# amounts_train = [331,1141,231,529,27,70,1257]
amounts_train = [300,300,300,300,27,70,300]
amounts_val = [82, 531, 104, 278, 6, 17, 4359]

inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te = replicate_data(X, Y, 'seven', amounts_train, amounts_val, seed_val)

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
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(train_loss_all, label='Train Loss')
    ax.plot(val_loss_all, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(outfile+'_loss.png')
    plt.close()
    
    # plotting Confusion Matrix and saving
    fig, ax = plt.subplots(3,1, figsize=(12,20))
    ConfusionMatrixDisplay.from_predictions(train_truth_values, train_predictions, ax = ax[0], normalize='true', cmap=cm_blues, display_labels=custom_labs, colorbar=False)
    ConfusionMatrixDisplay.from_predictions(val_truth_values, val_predictions, ax = ax[1], normalize='true', cmap=cm_blues, display_labels=custom_labs, colorbar=False)
    ConfusionMatrixDisplay.from_predictions(test_truth_values, test_predictions, ax = ax[2], normalize='true', cmap=cm_blues, display_labels=custom_labs, colorbar=False)
    
    # setting plotting attributes here
    ax[0].set_title('Training Set')
    ax[1].set_title('Validation Set')
    ax[2].set_title('Testing Set')
    plt.tight_layout()
    plt.savefig(outfile+'_CM.png')
    plt.close()
    
    # printout for classifcation report for all sets
    print('Target Set Report')
    print(classification_report(train_truth_values, train_predictions, target_names=custom_labs, zero_division=0))
    print('Validation Set Report')
    print(classification_report(val_truth_values, val_predictions, target_names=custom_labs, zero_division=0))
    print('Testing Set Report')
    print(classification_report(test_truth_values, test_predictions, target_names=custom_labs, zero_division=0))

    # saving the network settings
    torch.save(NetInstance.state_dict(),outfile+'_Settings')

if __name__ == '__main__':

    momentum_vals = [0.6,0.75, 0.9]
    learning_rate_vals = [4e-3, 4e-2, 4e-4, 4e-5, 4e-1]
    batch_size = [25,100,200]
    epochs = 3000
    filepath = "MLP_Runs_Results/Two_Layer/"  
    # outfile = filepath+"TwoLayers_300s_Mo09_30kepochs_lr4e1"
    outfile = "test"


    # Two Layer MLP
    TwoNN = TwoLayerMLP(8, 20, 7, weight_initialize=True)
    ## load settings in
    # loadpath = filepath+"test_Settings"
    # TwoNN.load_state_dict(torch.load(loadpath, map_location=device))
    optimizer = optim.SGD(TwoNN.parameters(), lr=learning_rate_vals[4], momentum=momentum_vals[2])
    main(epochs,TwoNN,optimizer,outfile)

    # file name for the following loop
    # outfile = [ filepath+'TwoLayers_300s_Mo09_10kepochs_lr4e3',
    #             filepath+'TwoLayers_300s_Mo09_10kepochs_lr4e2',
    #             filepath+'TwoLayers_300s_Mo09_10kepochs_lr4e4',
    #             filepath+'TwoLayers_300s_Mo09_10kepochs_lr4e5',
    #             filepath+'TwoLayers_300s_Mo09_10kepochs_lr4e1']
    # for i, learning_rate in enumerate(learning_rate_vals):
    #     TwoNN = TwoLayerMLP(8, 20, 7, weight_initialize=True)
    #     optimizer = optim.SGD(TwoNN.parameters(), lr=learning_rate, momentum=momentum_vals[2])
    #     main(epochs,TwoNN,optimizer,outfile[i])

    # file name for the following loop
    # outfile = [filepath+'TwoLayers_300s_Mo06_10kepochs_lr4e1',\
    #     filepath+'TwoLayers_300s_Mo075_10kepochs_lr4e1', filepath+'TwoLayers_300s_Mo09_10kepochs_lr4e1']
    # for i, momentum in enumerate(momentum_vals):
    #     TwoNN = TwoLayerMLP(8, 20, 7)
    #     optimizer = optim.SGD(TwoNN.parameters(), lr=learning_rate_vals[4], momentum=momentum)
    #     main(epochs,TwoNN,optimizer,outfile[i])








    # Single Layer MLP
    # BaseNN = BaseMLP(8, 20, 7, weight_initialize=True)

    ## load settings in
    # loadpath = filepath+"300s_Split_Mo09_30kepochs_lr4e2_Settings"
    # BaseNN.load_state_dict(torch.load(loadpath, map_location=device))

    # optimizer = optim.SGD(BaseNN.parameters(), lr=learning_rate_vals[0], momentum=momentum_vals[1])
    
    # setting scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000], gamma=0.1, verbose=False)
  
    # main(epochs,BaseNN,optimizer,outfile)






    # file name for the following loop
    # outfile = [ '300s_Split_Mo09_10kepochs_lr4e3',
    #             '300s_Split_Mo09_10kepochs_lr4e2',
    #             '300s_Split_Mo09_10kepochs_lr4e4',
    #             '300s_Split_Mo09_10kepochs_lr4e5']
    # for i, learning_rate in enumerate(learning_rate_vals):
    #     BaseNN = BaseMLP(8, 20, 7, weight_initialize=True)
    #     optimizer = optim.SGD(BaseNN.parameters(), lr=learning_rate, momentum=momentum_vals[2])
    #     main(epochs,BaseNN,optimizer,outfile[i])

    # file name for the following loop
    #outfile = ['LR4e-3_B25_E10k_M06', 'LR4e-3_B25_E10k_M075', 'LR4e-3_B25_E10k_M09']
    #for i, momentum in enumerate(momentum_vals):
        #BaseNN = BaseMLP(8, 20, 3)
        #optimizer = optim.SGD(BaseNN.parameters(), lr=learning_rate_vals[0], momentum=momentum)
        #main(epochs,BaseNN,optimizer,outfile[i])

toc = time.perf_counter()
print(f"Completed in {toc - tic:0.1f} seconds")