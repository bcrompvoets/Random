import time

tic = time.perf_counter()
# Check to make sure running on M1, will say 'arm'
import platform
print(platform.processor())

from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

from custom_dataloader import replicate_data_ind
from NN_Defs import CNN, validate, train

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f'Running on : {device}')

# YSO_EG_Stars
seds = np.load("SED_filenames.npy")
tars = np.load("Targets_SEDs.npy")[:,0]
filepath = 'SEDs/'
SEDs = []
for sed in seds:
    SEDs.append(filepath+sed)
SEDs = np.array(SEDs)

# YSO, EG, and Stars
train_amount = [1500,1500,1500]
valid_amount = [665,440,4715]

ind_tr, ind_va, ind_te = replicate_data_ind(tars, train_amount, valid_amount)
SEDs_tr = SEDs[ind_tr]
SEDs_va = SEDs[ind_va]
SEDs_te = SEDs[ind_te]

tar_tr = np.array(tars[ind_tr])
tar_va = np.array(tars[ind_va])
tar_te = np.array(tars[ind_te])


transformer = transforms.ToTensor()

inp_tr = []
inp_va = []
inp_te = []

for SED in SEDs_tr:
    inp_tr.append(transformer(Image.open(SED).convert('L')))
for SED in SEDs_va:
    inp_va.append(transformer(Image.open(SED).convert('L')))
for SED in SEDs_te:
    inp_te.append(transformer(Image.open(SED).convert('L')))



inp_tr = torch.stack(list(inp_tr))
inp_va = torch.stack(list(inp_va))
inp_te = torch.stack(list(inp_te))

# print(inp_tr.size())

inp_tr = torch.as_tensor(inp_tr)
tar_tr = torch.as_tensor(tar_tr,dtype=torch.long)
inp_va = torch.as_tensor(inp_va)
tar_va = torch.as_tensor(tar_va,dtype=torch.long)
inp_te = torch.as_tensor(inp_te)
tar_te = torch.as_tensor(tar_te,dtype=torch.long)

# pass tensors into TensorDataset instances
train_data = data_utils.TensorDataset(inp_tr, tar_tr)
val_data = data_utils.TensorDataset(inp_va, tar_va)
test_data = data_utils.TensorDataset(inp_te, tar_te)

# constructing data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=25, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=25, shuffle=True)


custom_labs = ['YSO','EG','Star']
# custom_labs = ['Class I', 'Class II', 'Flat-Spectrum', 'Class III']




if __name__ == '__main__':
    # choose your loss.
    criterion = nn.CrossEntropyLoss()
    num_classes = 3
    num_epochs = 100
    learning_rate = 0.01

    # use_gpu = torch.cuda.is_available()
    device = torch.device("cpu")
    model = CNN(num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
        
    # Run the training loop over the epochs (evaluate after each)

    loss_tr = []
    loss_va = []

    for epoch in range(1, num_epochs+ 1):
        train_loss, pred_tr, tar_tr = train(epoch, model, optimizer, train_loader, device)
        val_loss, pred_va, tar_va = validate(model,val_loader,device)
        loss_tr.append(train_loss)
        loss_va.append(val_loss)
    
    outfile= 'test2'
    # plotting losses and saving fig
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(loss_tr, label='Train Loss')
    ax.plot(loss_va, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(outfile+'_loss.png')
    plt.close()

    print(classification_report(tar_va,pred_va))

    # momentum_vals = [0.6,0.75, 0.9]
    # learning_rate_vals = [1e-1, 1e-2, 1e-3]
    # epochs = 3000
    # filepath = "MLP_Runs_Results/YSO/"
    # filepaths = [filepath+"OneLayer/", filepath+"TwoLayer/", filepath+"FiveLayer/"]


    # with mp.Pool(6) as pool:
    #     bestfiles = pool.starmap(find_best_MLP, iters)

    # print(bestfiles) 


toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")