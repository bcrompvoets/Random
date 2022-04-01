import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def replicate_data(inputs, targets, output, amounts_train, amounts_val, seed):
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
    amounts_train : list of ints
        How much of each label to grab for the training set. Length must match amount of unique labels of targets.
    amounts_val : list of ints
        How much of each label to grab for the validation set. Leftover amounts are passed to the testing set.
    seed : int
        Value passed to rng.default_rng() call, as well as shuffle() call.
    """    

    # setting default_rng with seed value
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
    train_indexes = [indexes[i][0:amount] for i, amount in enumerate(amounts_train)]
    val_indexes = [indexes[i][amount1:amount1+amount2] for i, (amount1, amount2) in enumerate(zip(amounts_train, amounts_val))]
    test_indexes = [indexes[i][amount1+amount2:] for i, (amount1, amount2) in enumerate(zip(amounts_train, amounts_val))]

    # concatenate arrays into 1D
    train_indexes = np.concatenate(train_indexes)
    val_indexes = np.concatenate(val_indexes)
    test_indexes = np.concatenate(test_indexes)

    # building new dataframes
    train_df = data.iloc[train_indexes]
    val_df = data.iloc[val_indexes]
    test_df = data.iloc[test_indexes]

    # enter here if user needs three labels
    if output  == 'three':
        # need to re-label [2-6] to just [2]
        for i in range(2,7):
            train_df.loc[train_df['Label'] == i, 'Label'] = 2
            val_df.loc[val_df['Label'] == i, 'Label'] = 2
            test_df.loc[test_df['Label'] == i, 'Label'] = 2

    # extract out new inputs and new targets
    train_targets = train_df.pop('Label').to_numpy()
    train_inputs = train_df.to_numpy()

    val_targets = val_df.pop('Label').to_numpy()
    val_inputs = val_df.to_numpy()

    test_targets = test_df.pop('Label').to_numpy()
    test_inputs = test_df.to_numpy()

    # since labels are technically ordered, shuffle before export
    # SHUFFLE WHILE CONSERVING INPUT-LABEL PAIRS
    train_inputs, train_targets = shuffle(train_inputs, train_targets, random_state=seed)
    val_inputs, val_targets = shuffle(val_inputs, val_targets, random_state=seed)
    test_inputs, test_targets = shuffle(test_inputs, test_targets, random_state=seed)

    return [train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets]
