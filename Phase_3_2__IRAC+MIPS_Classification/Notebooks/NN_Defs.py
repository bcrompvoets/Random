# import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
from custom_dataloader import replicate_data
import random
import time
import matplotlib.pyplot as plt
from numba import jit
import warnings
warnings.filterwarnings('ignore')

# to get number of parameters in network
def get_n_params(model):
    """Function to compute the number of free parameters in a model.

    Parameters
    ----------
    model : nn.Module subclass
        class built from inheriting nn.Module

    Returns
    -------
    int
        number of parameters
    """    
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

def train(epoch, model, optimizer, train_loader, device):
    """Training loop for network.

    - Uses CrossEntropy for loss function
    """    
    model.to(device) # send to device

    # setting export variables
    train_loss = []
    predictions = []
    truth_values = []

    #print('Number of parameters: {}'.format(get_n_params(model)))

    model.train() # setting model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        loss = F.cross_entropy(output, target.squeeze(-1).long())
        loss.backward()
        optimizer.step()

        # store training loss
        train_loss.append(loss.item())

        # storing training predictions and truth values
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        predictions.append(pred.squeeze(-1).cpu().numpy())
        truth_values.append(target.squeeze(-1).cpu().numpy())

    # average the training loss list to export one value here
    average_train_loss = np.mean(np.array(train_loss))


    # changing the predictions and truth values to flat arrays
    predictions = np.concatenate(predictions, axis=0)
    truth_values = np.concatenate(truth_values, axis=0)
        
    return average_train_loss, predictions, truth_values


            
def validate(model, val_loader, device):
    """Validation loop for network.""" 
    
    model.to(device) # send to device
    
    # setting in evaluation mode will iterate through entire dataloader according to batch size
    model.eval() # set network into evaluation mode
    
    # setting tracker variables
    val_loss = 0
    correct = 0

    # set export variables
    accuracy_list = []
    predictions = []
    truth_values = []

    # start looping through the batches
    for i, (data, target) in enumerate(val_loader):
        # send to device
        data, target = data.to(device), target.to(device)

        output = model(data.float())
        val_loss += F.cross_entropy(output, target.squeeze(-1).long(), reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability    
        
        # storing predictions and truth values
        predictions.append(pred.squeeze(-1).cpu().numpy())
        truth_values.append(target.squeeze(-1).cpu().numpy())
        
    # computing losses 
    val_loss /= len(val_loader.dataset)
    
    # changing the predictions and truth values to flat arrays
    predictions = np.concatenate(predictions, axis=0)
    truth_values = np.concatenate(truth_values, axis=0)

    # returning statement with all needed quantities
    return val_loss, predictions, truth_values

            
def test(model, test_tensor, device):
    """Test loop for network.""" 
    
    model.to(device) # send to device
    
    # setting in evaluation mode will iterate through entire dataloader according to batch size
    model.eval() # set network into evaluation mode
    val_loss = 0
    predictions = []

    # start looping through the batches
    for i, (data,target) in enumerate(test_tensor):
        # send to device
        data = data.to(device)
        target = target.to(device)

        output = model(data.float())
        val_loss += F.cross_entropy(output, target.squeeze(-1).long(), reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability    
        
        # storing predictions and truth values
        predictions.append(pred.squeeze(-1).cpu().numpy())
    
    # changing the predictions and truth values to flat arrays
    predictions = np.concatenate(predictions, axis=0)

    # returning statement with all needed quantities
    return predictions

class BaseMLP(nn.Module):
    """MLP with only one hidden layer"""
    
    def __init__(self, input_size, n_hidden, output_size, weight_initialize=True):
        super(BaseMLP, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.output_size)
        
        # weight initialization here
        if weight_initialize:
            # for fc1 layer
            torch.nn.init.uniform_(self.fc1.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc2 layer
            torch.nn.init.uniform_(self.fc2.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class TwoLayerMLP(nn.Module):
    """ Two Layer MLP"""
    
    def __init__(self, input_size, n_hidden, output_size, weight_initialize=True):
        super(TwoLayerMLP, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden, self.output_size)
        
        # weight initialization here
        if weight_initialize:
            # for fc1 layer
            torch.nn.init.uniform_(self.fc1.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc2 layer
            torch.nn.init.uniform_(self.fc2.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc3 layer
            torch.nn.init.uniform_(self.fc3.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1) 
        return x

class FiveLayerMLP(nn.Module):
    """ Two Layer MLP"""
    
    def __init__(self, input_size, n_hidden, output_size, weight_initialize=True):
        super(FiveLayerMLP, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc4 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc5 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc6 = nn.Linear(self.n_hidden, self.output_size)
        
        # weight initialization here
        if weight_initialize:
            # for fc1 layer
            torch.nn.init.uniform_(self.fc1.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc2 layer
            torch.nn.init.uniform_(self.fc2.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc3 layer
            torch.nn.init.uniform_(self.fc3.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc4 layer
            torch.nn.init.uniform_(self.fc4.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc5 layer
            torch.nn.init.uniform_(self.fc5.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
            # for fc6 layer
            torch.nn.init.uniform_(self.fc6.weight, -1/np.sqrt(input_size), 1/np.sqrt(input_size))
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        x = self.fc5(x)
        x = torch.sigmoid(x)
        x = self.fc6(x)
        x = F.softmax(x, dim=1)
        return x


def MLP_data_setup(inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te):
            
    # scaling data according to training inputs
    scaler_S = StandardScaler().fit(inp_tr)
    inp_tr = scaler_S.transform(inp_tr)
    inp_va = scaler_S.transform(inp_va)
    inp_te = scaler_S.transform(inp_te) 

    # creation of tensor instances

    inp_tr = torch.as_tensor(inp_tr)
    tar_tr = torch.as_tensor(tar_tr)
    inp_va = torch.as_tensor(inp_va)
    tar_va = torch.as_tensor(tar_va)
    inp_te = torch.as_tensor(inp_te)
    tar_te = torch.as_tensor(tar_te)

    # pass tensors into TensorDataset instances
    train_data = data_utils.TensorDataset(inp_tr, tar_tr)
    val_data = data_utils.TensorDataset(inp_va, tar_va)
    test_data = data_utils.TensorDataset(inp_te, tar_te)

    # constructing data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False) # Changed shuffle to false to see if we can retrieve the original set after predicting
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader

def bootstrap(NN, epochs, OptInstance, X, Y, num_train, num_val, device, ScheduleInstance=None):
    train_loader, val_loader = MLP_data_setup(X, Y, num_train, num_val)
    for epoch in range(0, epochs):
        train_loss, pred_tar, tar_tr = train(epoch, NN, OptInstance, train_loader, device)
        val_loss, pred_va, tar_va = validate(NN, val_loader, device)
        
        if ScheduleInstance is not None:
            ScheduleInstance.step()

    ScoresA = accuracy_score(tar_va,pred_va)
    ScoresR = recall_score(tar_va,pred_va,average=None,zero_division=1)
    ScoresP = precision_score(tar_va,pred_va,average=None,zero_division=1)

    return ScoresR, ScoresP, ScoresA

@jit
def find_best_MLP(MLP, filepath_to_MLPdir, learning_rate_vals, momentum_vals, train_loader, val_loader, test_loader, custom_labs, device):
    f1Max = 0.5
    tic1 = time.perf_counter()
    for lr in learning_rate_vals:
        for mo in momentum_vals:
            for n in [10,20,50]:
                outfile = filepath_to_MLPdir + "LR_" + str(lr) + "_MO_" + str(mo) + "_NEUR_" + str(n)
                NN = MLP(11,n,len(custom_labs))
                # load path
                # loadpath = filepath_to_MLPdir + "LR_" + str(lr) + "_MO_" + str(mo) + "_NEUR_" + str(n) +"_Settings"
                # NN.load_state_dict(torch.load(loadpath, map_location=device))
                optimizer = optim.SGD(NN.parameters(), lr=lr, momentum=mo)
                f1score = main(3000, NN, optimizer, outfile, train_loader, val_loader, test_loader, custom_labs, device)
                if f1score[0] > f1Max and f1score[1] != 0 and f1score[2] != 0:
                    f1Max = f1score[0]
                    bestfile = outfile
    toc1 = time.perf_counter()
    timed = (toc1 - tic1)/60
    print(f"Completed full MLP in {timed} minutes")
    print(f'{bestfile}\n')
    return bestfile

def main(epochs, NetInstance, OptInstance, outfile, train_loader, val_loader, test_loader, custom_labs, device, ScheduleInstance=None):

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
    
    
    with open(outfile+'_Classification_Reports.txt','w') as f:
        f.write('Target Set Report (c2d CLOUDS)\n')
        f.write(classification_report(train_truth_values, train_predictions, target_names=custom_labs, zero_division=0))
        f.write('\nValidation Set Report (NGC 2264)\n')
        f.write(classification_report(val_truth_values, val_predictions, target_names=custom_labs, zero_division=0))
        if len(np.unique(test_predictions))!=len(custom_labs):
            print(f'\n Testing failed for {outfile} \n')
        else:
            f.write('\nTesting Set Report (c2d CLOUDS)\n')
            f.write(classification_report(test_truth_values, test_predictions, target_names=custom_labs, zero_division=0))
        

    # saving the network settings
    torch.save(NetInstance.state_dict(),outfile+'_Settings')

    return f1_score(val_truth_values,val_predictions,average=None,zero_division=1)