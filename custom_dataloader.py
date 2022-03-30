import numpy as np
import pandas as pd

def replicate_data(inputs, targets, output, amounts, seed):
    """Creates both training and testing datasets (targets and inputs), custom built to the
    amounts needed, given some input dataset.

    Parameters
    ----------
    inputs : np.ndarray
        Input dataset
    targets : np.ndarray
        Targets associated with the inputs
    output : str
        'three' will change the function such that only 3 labels are returned
    amounts : list of ints
        How much of each label to grab for the training set. Length must be 7.
    seed : int
        Value passed to rng.default_rng() call. 
    """    

    rng = np.random.default_rng(seed = seed)

    # convert inputs and targets to dataframe
    data = pd.DataFrame(inputs)
    data['Label'] = targets

    # separate all label types
    indexes = [data[data['Label'] == i].index.to_numpy() for i in range(7)]

    # shuffle indexes
    for i in indexes:
        rng.shuffle(i)

    # build training and test sets
    train_indexes = [indexes[i][0:amount] for i, amount in enumerate(amounts)]
    test_indexes = [indexes[i][amount:] for i, amount in enumerate(amounts)]

    # concatenate arrays into 1D
    train_indexes = np.concatenate(train_indexes)
    test_indexes = np.concatenate(test_indexes)

    # building new dataframe

    train_df = data.iloc[train_indexes]
    test_df = data.drop(train_indexes)

    # enter here if user needs three labels
    if output  == 'three':
        # need to re-label [2-6] to just [2]
        for i in range(2,7):
            train_df.loc[train_df['Label'] == i, 'Label'] = 2
            test_df.loc[test_df['Label'] == i, 'Label'] = 2

    # extract out new inputs and new targets
    train_targets = train_df.pop('Label').to_numpy()
    train_inputs = train_df.to_numpy()

    test_targets = test_df.pop('Label').to_numpy()
    test_inputs = test_df.to_numpy()

    return train_inputs, train_targets, test_inputs, test_targets
