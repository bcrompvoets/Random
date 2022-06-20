# library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import joblib
# custom script inputs
from NN_Defs import test, TwoLayerMLP, validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

device = torch.device("cpu")
print(f'Running on : {device}')

inputs = np.load("../Data_and_Results/Inputs_YSO_EG_Stars_alpha.npy")
targets = np.load("../Data_and_Results/Targets_YSO_EG_Stars_alpha.npy") #Include for now

# scaling data according to training inputs
scaler_S = StandardScaler().fit(inputs)
inputs = scaler_S.transform(inputs)
# creation of tensor instances

#Modify inputs to not contain alpha values
inputs_ALL = inputs.transpose()[slice(8)]
inputs_ALL = inputs_ALL.transpose()

inputs_tens = torch.as_tensor(inputs_ALL)
targets_tens = torch.as_tensor(targets.ravel())

# # pass tensors into TensorDataset instances
inputs_data = data_utils.TensorDataset(inputs_tens, targets_tens)

inputs_loader = torch.utils.data.DataLoader(inputs_data, batch_size=25, shuffle=True)
# create nn instance
NN = TwoLayerMLP(8, 50, 3)
# load in saved state of network
loadpath = '../MLP_Runs_Results/YSO_EG_Stars/CLIII_F_as_Stars_Best_Results/Even/TwoLayer/LR_0.1_MO_0.75_NEUR_50_Settings'
NN.load_state_dict(torch.load(loadpath, map_location=device))

# compute predictions for if YSO, EG, or Stars
preds = test(NN, inputs_loader, device)

#Compute for which YSO it is
inputs_YSO = inputs[np.where(preds==0)[0]]
rf = joblib.load("../Data_and_Results/YSO_RF_alpha_Settings.joblib")
YSO_preds = rf.predict(inputs_YSO)

# Plot alpha separation for YSOs
bins = np.linspace(-3, 6, 45)
aIII = np.where(YSO_preds==3.)
plt.hist(inputs[aIII].transpose()[8],bins,histtype='step',label='Class III')
aII = np.where(YSO_preds==2.)
plt.hist(inputs[aII].transpose()[8],bins,histtype='step',label='Class II')
aFS = np.where(YSO_preds==1.)
plt.hist(inputs[aFS].transpose()[8],bins,histtype='step',label='Class FS')
aI = np.where(YSO_preds==0.)
plt.hist(inputs[aI].transpose()[8],bins,histtype='step',label='Class I')
plt.axvline(x=-1.6,color='k', linestyle='--')
plt.axvline(x=-0.3,color='k', linestyle='--')
plt.axvline(x=0.3,color='k', linestyle='--')
plt.legend()
plt.xlabel('Spectral Index α')
plt.title('YSO Spectral Index Histogram')
plt.savefig("../Data_and_Results/Trial1.png")




with open("Program_Results_Trial1", "w") as f1:
    f1.write("Predictions for Megeath/Rapson \n \n YSO EG Stars \n")
    f1.write(classification_report(targets,preds,target_names=["YSO", "EG", "Stars"]))