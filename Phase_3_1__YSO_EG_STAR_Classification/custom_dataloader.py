import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random

def replicate_data(inputs, targets, amounts_train, amounts_valid):
    """Creates both training, validation, and testing datasets (targets and inputs), custom built to the
    amounts needed, given some input dataset.

    Parameters
    ----------
    inputs : np.ndarray
        Input dataset
    targets : np.ndarray
        Targets associated with the inputs
    amounts_train: list of ints
        How much of each label to grab for the training set. Length must match amount of unique labels of targets.
    amounts_val : list of ints
        How much of each label to grab for the validation set. Leftover amounts are passed to the testing set.
    """ 
    
    if len(amounts_train) != 3:
        print("Length of amounts_train does not match number of classes in targets.")
        raise ValueError
    
    # Fetch indices of each class
    YSO_index = np.where(targets==0)[0]
    EG_index = np.where(targets==1)[0]
    Sta_index = np.where(targets==2)[0]
    
    # These arrays will hold the indices of the shuffled indices
    train_indices = []
    valid_indices = []
    test_indices = []

    #Randomly choose amounts_train and amounts_valid amounts from these three classes
    for i, type_in in enumerate([YSO_index, EG_index, Sta_index]):
        # Shuffle the array of indices 
        type_in = shuffle(type_in,random_state=random.randint(0,1000))
        
        # Take the first amount of this shuffled set as the training set, the next amount for validation
        train_indices = np.append(train_indices,type_in[0:amounts_train[i]]).astype(int)
        valid_indices = np.append(valid_indices,type_in[amounts_train[i]+1:amounts_train[i]+amounts_valid[i]]).astype(int)
        test_indices = np.append(test_indices,type_in[amounts_train[i]+amounts_valid[i]:-1]).astype(int)
        

    # Create arrays that will hold the actual values for each case
    train_input = inputs[train_indices]
    train_target = targets[train_indices]
    valid_input = inputs[valid_indices]
    valid_target = targets[valid_indices]
    test_input = inputs[test_indices]
    test_target = targets[test_indices]

    return train_input, train_target, valid_input, valid_target, test_input,test_target
