import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import torch
import torch.utils.data as data_utils

from NN_Defs import TwoLayerMLP, validate, MLP_data_setup, test
from custom_dataloader import replicate_data_single


device = torch.device("cpu")
outfile = "NGC2264_MPvsIR_1k_Final.txt"

# File to use to scale rest of data
file_tr = "../Data/c2d_1k_INP.npy" 
file_tr_tar = "../Data/c2d_1k_TAR.npy" 
X_tr = np.load(file_tr) # Load input data
Y_tr = np.load(file_tr_tar) # Load target data
X_tr = np.float32(X_tr)
Y_tr = np.float32(Y_tr)
X_tr = np.c_[ 0:len(X_tr), X_tr ] # Add an index for now
inp_tr, tar_tr = replicate_data_single(X_tr, Y_tr, [len(np.where(Y_tr==0.)[0]),len(np.where(Y_tr==1.)[0]),len(np.where(Y_tr==2.)[0])])
ind_tr = inp_tr[:,0] # Create array which only holds indices
inp_tr = inp_tr[:,1:] # Remove index from input

# YSO_EG_Stars Test
X_te = np.load("../Data/NGC2264_INP.npy") # Load input data
Y_te = np.load("../Data/NGC2264_TAR.npy") # Load target data
X_te = np.float32(X_te)
Y_te = np.float32(Y_te)
X_te = np.c_[ 0:len(X_te), X_te ] # Add an index for now
inp_te, tar_te = replicate_data_single(X_te, Y_te, [len(np.where(Y_te==0.)[0]),len(np.where(Y_te==1.)[0]),len(np.where(Y_te==2.)[0])])
ind_te = inp_te[:,0] # Create array which only holds indices
inp_te = inp_te[:,1:] # Remove index from input


# REMOVE MIPS DATA FOR IRAC ONLY
inp_TR = np.delete(inp_tr,np.s_[8:10],axis=1)
inp_TE = np.delete(inp_te,np.s_[8:10],axis=1)

IR_train, IR_valid, IR_test = MLP_data_setup(inp_TR, tar_tr,inp_TE, tar_te, inp_TE, tar_te)


# IRAC + MIPS REMOVE -99 DATA
MP_inp_tr = inp_tr[np.where(inp_tr[:,9]!=-99)[0]]
MP_tar_tr = tar_tr[np.where(inp_tr[:,9]!=-99)[0]]
MP_inp_te = inp_te[np.where(inp_te[:,9]!=-99)[0]]
MP_tar_te = tar_te[np.where(inp_te[:,9]!=-99)[0]]

MP_train, MP_valid, MP_test = MLP_data_setup(MP_inp_tr, MP_tar_tr, MP_inp_te, MP_tar_te, MP_inp_te, MP_tar_te)



# IRAC only
NN_IR = TwoLayerMLP(9, 20, 3)
NN_IR.load_state_dict(torch.load("../MLP_Settings/IRAC_TwoLayer_LR_0.001_MO_0.9_NEUR_20_Settings", map_location=device))

IR_preds_tr = test(NN_IR, IR_train, device)
IR_preds_te = test(NN_IR, IR_test, device)

#IRAC+MIPS
NN = TwoLayerMLP(11, 10, 3)
NN.load_state_dict(torch.load("../MLP_Settings/MIPS_TwoLayer_LR_0.01_MO_0.9_NEUR_10_Settings", map_location=device))

MP_preds_tr = test(NN, MP_train, device)
MP_preds_te = test(NN, MP_test, device)

#Combination
IR_tr = pd.DataFrame(data={"IRAC only Pred": IR_preds_tr},index = ind_tr).sort_index()
MP_tr = pd.DataFrame(data={"IRAC+MIPS Pred": MP_preds_tr},index = ind_tr[np.where(inp_tr[:,9]!=-99)[0]]).sort_index()
TAR_tr = pd.DataFrame(data={"Truth": tar_tr.ravel()},index = ind_tr).sort_index()

comb_df = IR_tr.join(MP_tr)
comb_df = comb_df.join(TAR_tr)
COMB_tr = []
for i, a in enumerate(comb_df["IRAC+MIPS Pred"].values):
    if a==2:
        COMB_tr.append(a)
    else: 
        COMB_tr.append(comb_df["IRAC only Pred"].values[i])


IR_te = pd.DataFrame(data={"IRAC only Pred": IR_preds_te},index = ind_te).sort_index()
MP_te = pd.DataFrame(data={"IRAC+MIPS Pred": MP_preds_te},index = ind_te[np.where(inp_te[:,9]!=-99)[0]]).sort_index()
TAR_te = pd.DataFrame(data={"Truth": tar_te.ravel()},index = ind_te).sort_index()

comb_df = IR_te.join(MP_te)
comb_df = comb_df.join(TAR_te)
COMB_te = []
for i, a in enumerate(comb_df["IRAC+MIPS Pred"].values):
    if a==2:
        COMB_te.append(a)
    else: 
        COMB_te.append(comb_df["IRAC only Pred"].values[i])




with open("../Results/"+outfile,"w") as f:
    f.write("Network trained only on IRAC bands\n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,IR_preds_tr))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,IR_preds_te))
    f.write("\nNetwork trained on IRAC + MIPS 24μ bands\n Training data (c2d Survey)\n")
    f.write(classification_report(MP_tar_tr,MP_preds_tr))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(MP_tar_te,MP_preds_te))
    f.write("\nCombining the strongest classifications\n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,COMB_tr))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,COMB_te))