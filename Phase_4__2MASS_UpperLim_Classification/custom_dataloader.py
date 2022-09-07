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
    
    if len(amounts_train) != len(np.unique(targets)):
        print(f"Number of classes in targets is {len(np.unique(targets))}")
        print("Length of amounts_train does not match number of classes in targets.")
        raise ValueError
    
    class_indices = []
    for i in np.unique(targets):
        if amounts_train[i] > len(np.where(targets==i)[0]):
            print(f"Number of Class {i} in targets is {len(np.where(targets==i)[0])}")
            print("Amount specified is greater than amount available of this class.")
        class_indices.append(np.where(targets==i)[0])

    # These arrays will hold the indices of the shuffled indices
    train_indices = []
    valid_indices = []
    test_indices = []

    #Randomly choose amounts_train and amounts_valid amounts from these three classes
    for i, type_in in enumerate(class_indices):
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



def replicate_data_ind(targets, amounts_train, amounts_valid):
    """Creates both training, validation, and testing datasets (targets and inputs), custom built to the
    amounts needed, given some input dataset.

    Parameters
    ----------
    targets : np.ndarray
        Targets of the list you would like indices from
    amounts_train: list of ints
        How much of each label to grab for the training set. Length must match amount of unique labels of targets.
    amounts_val : list of ints
        How much of each label to grab for the validation set. Leftover amounts are passed to the testing set.
    """ 
    
    if len(amounts_train) != len(np.unique(targets)):
        print(f"Number of classes in targets is {len(np.unique(targets))}")
        print("Length of amounts_train does not match number of classes in targets.")
        raise ValueError
    
    class_indices = []
    for i in np.unique(targets):
        if amounts_train[i] > len(np.where(targets==i)[0]):
            print(f"Number of Class {i} in targets is {len(np.where(targets==i)[0])}")
            print("Amount specified is greater than amount available of this class.")
        class_indices.append(np.where(targets==i)[0])

    # These arrays will hold the indices of the shuffled indices
    train_indices = []
    valid_indices = []
    test_indices = []

    #Randomly choose amounts_train and amounts_valid amounts from these three classes
    for i, type_in in enumerate(class_indices):
        # Shuffle the array of indices 
        type_in = shuffle(type_in,random_state=random.randint(0,1000))
        
        # Take the first amount of this shuffled set as the training set, the next amount for validation
        train_indices = np.append(train_indices,type_in[0:amounts_train[i]]).astype(int)
        valid_indices = np.append(valid_indices,type_in[amounts_train[i]+1:amounts_train[i]+amounts_valid[i]]).astype(int)
        test_indices = np.append(test_indices,type_in[amounts_train[i]+amounts_valid[i]:-1]).astype(int)

    return train_indices, valid_indices, test_indices


def replicate_data_single(inputs, targets, amounts):
    """Creates custom dataset (targets and inputs), custom built to the
    amounts needed, given some input dataset.

    Parameters
    ----------
    inputs : np.ndarray
        Input dataset
    targets : np.ndarray
        Targets associated with the inputs
    amounts: list of ints
        How much of each label to grab for the set. 
    """ 
    
    if len(amounts) != len(np.unique(targets)):
        print(f"Number of classes in targets is {len(np.unique(targets))}")
        print("Length of amounts_train does not match number of classes in targets.")
        raise ValueError
    
    class_indices = []
    for i in np.unique(targets):
        if amounts[int(i)] > len(np.where(targets==i)[0]):
            print(f"Number of Class {i} in targets is {len(np.where(targets==i)[0])}")
            print("Amount specified is greater than amount available of this class.")
            raise ValueError
        class_indices.append(np.where(targets==i)[0])

    # These arrays will hold the indices of the shuffled indices
    train_indices = []

    #Randomly choose amounts_train and amounts_valid amounts from these three classes
    for i, type_in in enumerate(class_indices):
        # Shuffle the array of indices 
        type_in = shuffle(type_in,random_state=random.randint(0,1000))
        
        # Take the first amount of this shuffled set as the training set, the next amount for validation
        train_indices = np.append(train_indices,type_in[0:amounts[i]]).astype(int)
  
    train_indices = shuffle(train_indices,random_state=random.randint(0,1000))

    # Create arrays that will hold the actual values for each case
    train_input = inputs[train_indices]
    train_target = targets[train_indices]

    return train_input, train_target
