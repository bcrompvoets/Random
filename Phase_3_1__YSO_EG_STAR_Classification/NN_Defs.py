# import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from custom_dataloader import replicate_data
import random


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

        # storing traning predictions and truth values
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

class BaseMLP(nn.Module):
    """ Base NN MLP Class from Cornu Paper"""
    
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


def MLP_data_setup(X,Y, amounts_train, amounts_val):
    inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te = replicate_data(X, Y, 'seven', amounts_train, amounts_val,random.randint(0,1000))
            
    # scaling data according to training inputs
    scaler_S = StandardScaler().fit(inp_tr)
    inp_tr = scaler_S.transform(inp_tr)
    inp_va = scaler_S.transform(inp_va)
    # inp_te = scaler_S.transform(inp_te) 

    # creation of tensor instances

    inp_tr = torch.as_tensor(inp_tr)
    tar_tr = torch.as_tensor(tar_tr)
    inp_va = torch.as_tensor(inp_va)
    tar_va = torch.as_tensor(tar_va)
    # inp_te = torch.as_tensor(inp_te)
    # tar_te = torch.as_tensor(tar_te)

    # pass tensors into TensorDataset instances
    train_data = data_utils.TensorDataset(inp_tr, tar_tr)
    val_data = data_utils.TensorDataset(inp_va, tar_va)
    # test_data = data_utils.TensorDataset(inp_te, tar_te)

    # constructing data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=25, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=25, shuffle=True)
    return train_loader, val_loader

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