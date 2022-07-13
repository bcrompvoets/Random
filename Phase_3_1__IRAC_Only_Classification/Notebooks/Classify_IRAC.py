# library imports
from ast import AST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import joblib
# custom script inputs
from NN_Defs import test, TwoLayerMLP, validate, MLP_data_setup
from custom_dataloader import replicate_data_single
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

device = torch.device("cpu")
print(f'Running on : {device}')

# YSO_EG_Stars Train
X_tr = np.load("../Data_and_Results/c2d_Inputs_CLOUDS_Train.npy") # Load input data
Y_tr = np.load("../Data_and_Results/c2d_Targets_CLOUDS_Train.npy") # Load target data
X_tr = np.float32(X_tr)
Y_tr = np.float32(Y_tr)
inp_tr, tar_tr = replicate_data_single(X_tr, Y_tr, [1082,1082,1082])

# YSO_EG_Stars Test
X_te = np.load("../Data_and_Results/Rap_Inputs_Test.npy") # Load input data
Y_te = np.load("../Data_and_Results/Rap_Targets_Test.npy") # Load target data
# X_te = np.load("../Data_and_Results/c2d_Inputs_CORES_Valid.npy") # Load input data
# Y_te = np.load("../Data_and_Results/c2d_Targets_CORES_Valid.npy") # Load target data
inp_te = np.float32(X_te)
tar_te = np.float32(Y_te)

scaler_S = StandardScaler().fit(inp_tr)
inp_tr = scaler_S.transform(inp_tr)
inp_te = scaler_S.transform(inp_te) 

# creation of tensor instances
inp_tr = torch.as_tensor(inp_tr)
tar_tr = torch.as_tensor(tar_tr)
inp_te = torch.as_tensor(inp_te)
tar_te = torch.as_tensor(tar_te)

# pass tensors into TensorDataset instances
train_data = data_utils.TensorDataset(inp_tr, tar_tr)
test_data = data_utils.TensorDataset(inp_te, tar_te)
# constructing data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# create nn instance
NN = TwoLayerMLP(9, 20, 3)
# load in saved state of network
loadpath = '../MLP_Runs_Results/YSO_EG_Stars/c2d_ALL_Synth_10k_Best_Results/TwoLayer_LR_0.001_MO_0.9_NEUR_20_Settings'
NN.load_state_dict(torch.load(loadpath, map_location=device))

# compute predictions from what the network was trained on
preds, targets, inputs = test(NN, train_loader, device)
# compute predictions for test
preds_te, targets_te, inputs_te = test(NN, test_loader, device)

# # print(inputs_te.shape)

# inp_tr_2 = inp_tr[np.where(preds==0)[0]].cpu().detach().numpy()
# tar_tr_2 = targets[np.where(preds==0)[0]]
# inp_te_2 = inp_te[np.where(preds_te==0)[0]].cpu().detach().numpy()
# tar_te_2 = targets_te[np.where(preds_te==0)[0]]

# # creation of tensor instances
# inp_tr_2 = torch.as_tensor(inp_tr_2)
# tar_tr_2 = torch.as_tensor(tar_tr_2)
# inp_te_2 = torch.as_tensor(inp_te_2)
# tar_te_2 = torch.as_tensor(tar_te_2)
# # pass tensors into TensorDataset instances
# train_data_2 = data_utils.TensorDataset(inp_tr_2, tar_tr_2)
# test_data_2 = data_utils.TensorDataset(inp_te_2, tar_te_2)
# # constructing data loaders
# train_loader = torch.utils.data.DataLoader(train_data_2, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data_2, batch_size=32, shuffle=True)

# # compute predictions from what the network was trained on
# preds_2, targets_2, inputs_2 = test(NN, train_loader, device)
# # compute predictions for test
# preds_te_2, targets_te_2, inputs_te_2 = test(NN, test_loader, device)

# ii=0
# for i in np.where(preds==0)[0]:
#     preds[i] = preds_2[ii]
#     inputs[i] = inputs_2[ii]
#     ii += 1

# ii=0
# for i in np.where(preds_te==0)[0]:
#     preds_te[i] = preds_te_2[ii]
#     inputs_te[i] = inputs_te_2[ii]
#     ii += 1


with open("Program_Results_Trial1", "w") as f1:
    f1.write("Predictions for CLOUDS (even split) \n \n YSO EG Stars \n")
    f1.write(classification_report(targets,preds,target_names=["YSO", "EG", "Stars"]))
    f1.write("\n Predictions for Rapson \n")
    f1.write(classification_report(targets_te,preds_te,target_names=["YSO", "EG", "Stars"]))

# Compute for which YSO it is
# inp_va = inp_te
inputs_YSO = inputs_te[np.where(preds_te==0)[0]]
targets_YSO = targets_te[np.where(preds_te==0)[0]]
# Categorize by alphas
alphas_YSO = inputs_YSO[:,-1]

# inputs_YSO = inputs_YSO[:,:-1]

aSTc = np.where(targets_YSO==2)
aIII = np.where((alphas_YSO<=-1.6) & (targets_YSO==0))
aII = np.where((alphas_YSO>-1.6) & (alphas_YSO<=-0.3)& (targets_YSO==0))
aFS = np.where((alphas_YSO<=0.3) & (alphas_YSO>-0.3)& (targets_YSO==0))
aI = np.where((alphas_YSO>0.3) & (targets_YSO==0))
aEGc = np.where(targets_YSO==1)
custom_labs = ['Class I', 'Flat-Spectrum', 'Class II',  'Star can', 'EG can']
inds = [aI, aFS, aII, aSTc, aEGc]
bins = np.linspace(-3, 6, 45)

for i, a in enumerate(inds):
    targets_YSO[a] = i
    if i!=3:
        plt.hist(alphas_YSO[a],bins,histtype='step',label=custom_labs[i])


# boostcl = GradientBoostingClassifier(criterion='friedman_mse',max_depth=5,max_features='log2',
#                 n_estimators=150,n_iter_no_change=5,subsample=1.0,warm_start=False)
# boostcl.fit(inputs_YSO,targets_YSO.ravel())
# YSO_preds = boostcl.predict(inputs_YSO)
# rfcl = RandomForestClassifier(class_weight='balanced',criterion='entropy',max_features='log2',n_estimators=50,oob_score=False)
rfcl = joblib.load("../Data_and_Results/YSO_RF_alpha_Settings.joblib")
# rfcl.fit(inputs_YSO,targets_YSO.ravel())
YSO_preds = rfcl.predict(inputs_YSO) #WARNING CHECK WHAT THE NUMBERS MATCH UP TO
print(np.unique(YSO_preds))

# Plot alpha separation for YSOs
# bins = np.linspace(-3, 6, 45)
# pEGc = np.where(YSO_preds==5.)
pEGc = np.where(YSO_preds==4.)
pSTc = np.where(YSO_preds==3.)
pII = np.where(YSO_preds==2.)
pFS = np.where(YSO_preds==1.)
pI = np.where(YSO_preds==0.)

p_inds = [pI, pFS, pII, pSTc, pEGc]
pred_labs = ['Class I Pred', 'Flat-Spectrum Pred', 'Class II Pred', 'Star can Pred', 'EG can Pred']
for i, p in enumerate(p_inds):
    if i!=3:
        plt.hist(alphas_YSO[p],bins,histtype='bar',label=pred_labs[i])

plt.axvline(x=-1.6,color='k', linestyle='--')
plt.axvline(x=-0.3,color='k', linestyle='--')
plt.axvline(x=0.3,color='k', linestyle='--')
plt.legend()
plt.xlabel('Spectral Index α')
plt.title('YSO Spectral Index Histogram')
plt.savefig("Trial1.png")


print(classification_report(targets_YSO,YSO_preds,target_names=custom_labs))
# print(f"Feature importances: {rfcl.feature_importances_}")