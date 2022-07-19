import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.utils.data as data_utils

from NN_Defs import TwoLayerMLP, validate, MLP_data_setup, test, preproc_yso
from custom_dataloader import replicate_data_single
import joblib

device = torch.device("cpu")
outfile = "../Results/FullScript_Classification_Report.txt"

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



# IRAC MLP
NN_IR = TwoLayerMLP(9, 20, 3)
NN_IR.load_state_dict(torch.load("../MLP_Settings/IRAC_TwoLayer_LR_0.001_MO_0.9_NEUR_20_Settings", map_location=device))

MLP_preds_tr = test(NN_IR, IR_train, device)
MLP_preds_te = test(NN_IR, IR_test, device)

#IRAC CM
# rfcl = joblib.load("../CM_Settings/YSE_RF_Settings.joblib")
rfcl = RandomForestClassifier(class_weight='balanced',criterion='entropy',max_features='log2',n_estimators=50,oob_score=False)
rfcl.fit(inp_TR,tar_tr.ravel())
RF_preds_tr = rfcl.predict(inp_TR)
RF_preds_te = rfcl.predict(inp_TE)

#Combination
# def flag_All(pred1,pred2):
#     flag = []
#     for i, p1 in enumerate(pred1):
#         if p1 == pred2[i]:
#             flag.append(0)
#         else:
#             flag.append(1)
#     return flag

# flags_YSE_tr = flag_All(MLP_preds_tr,RF_preds_tr)
# flags_YSE_te = flag_All(MLP_preds_te,RF_preds_te)


# Entering YSO classification
# df_tr = pd.DataFrame({"3.6mag": inp_tr[:,0],"e_3.6mag": inp_tr[:,1],"4.5mag": inp_tr[:,2],"e_4.5mag": inp_tr[:,3],\
#     "5.8mag": inp_tr[:,4],"e_5.8mag": inp_tr[:,5],"8mag": inp_tr[:,6],"e_8mag": inp_tr[:,7],\
#         "3.6mag": inp_tr[:,8],"e_3.6mag": inp_tr[:,9], "alpha":inp_tr[:,10], "Target": tar_tr.ravel(), \
#             "MLP Pred": MLP_preds_tr, "RF Pred": RF_preds_tr, "Flag": flags_YSE_tr}, index=ind_tr)

# df_tr_flagged = df_tr[(df_tr["Flag"]==1) | (df_tr["Target"]==0)]
# inp_tr_flagged = df_tr_flagged[["3.6mag","e_3.6mag","4.5mag","e_4.5mag","5.8mag","e_5.8mag","8mag","e_8mag","alpha"]].values.astype(float)
# MLP_preds_TR = df_tr_flagged[["MLP Pred"]].values.astype(int)
# RF_preds_TR = df_tr_flagged[["RF Pred"]].values.astype(int)
# tar_TR = df_tr_flagged[["Target"]].values.astype(int)

# df_te = pd.DataFrame({"3.6mag": inp_te[:,0],"e_3.6mag": inp_te[:,1],"4.5mag": inp_te[:,2],"e_4.5mag": inp_te[:,3],\
#     "5.8mag": inp_te[:,4],"e_5.8mag": inp_te[:,5],"8mag": inp_te[:,6],"e_8mag": inp_te[:,7],\
#         "3.6mag": inp_te[:,8],"e_3.6mag": inp_te[:,9], "alpha":inp_te[:,10],"Target": tar_te.ravel(),\
#              "MLP Pred": MLP_preds_te, "RF Pred": RF_preds_te, "Flag": flags_YSE_te}, index=ind_te)

# df_te_flagged = df_te[(df_te["Flag"]==1) | (df_te["Target"]==0)]
# inp_te_flagged = df_te_flagged[["3.6mag","e_3.6mag","4.5mag","e_4.5mag","5.8mag","e_5.8mag","8mag","e_8mag","alpha"]].values.astype(float)
# MLP_preds_TE = df_te_flagged[["MLP Pred"]].values.astype(int)
# RF_preds_TE = df_te_flagged[["RF Pred"]].values.astype(int)
# tar_TE = df_te_flagged[["Target"]].values.astype(int)

# # Change preds to match up with new scheme
preproc_yso(inp_tr[:,-1],MLP_preds_tr)
preproc_yso(inp_te[:,-1],MLP_preds_te)

preproc_yso(inp_tr[:,-1],RF_preds_tr)
preproc_yso(inp_te[:,-1],RF_preds_te)

preproc_yso(inp_tr[:,-1],tar_tr)
preproc_yso(inp_te[:,-1],tar_te)

# Classify into YSO types
# rfcl_YSO = joblib.load("../CM_Settings/YSO_RF_Settings.joblib")
rfcl_YSO = RandomForestClassifier(class_weight='balanced',criterion='entropy',max_features='log2',n_estimators=50,oob_score=False)
rfcl_YSO.fit(inp_TR,tar_tr.ravel())
YSO_preds_tr = rfcl_YSO.predict(inp_TR)
YSO_preds_te = rfcl_YSO.predict(inp_TE)
print(np.unique(YSO_preds_tr))
print(np.unique(YSO_preds_te))
def flag_YSO(pred1,pred2,pred3):
    flag = []
    for i, p3 in enumerate(pred3):
        if pred1[i]==pred2[i]:
            flag.append(0)
        elif pred1[i]==p3:
            flag.append(1)
        elif pred2[i]==p3:
            flag.append(2)
        else:
            flag.append(3)
    return flag

flags_YSO_tr = np.array(flag_YSO(MLP_preds_tr,RF_preds_tr,YSO_preds_tr))
flags_YSO_te = np.array(flag_YSO(MLP_preds_te,RF_preds_te,YSO_preds_te))

pred_tr = []
for i, flag in enumerate(flags_YSO_tr):
    if flag == 0:
        pred_tr.append(MLP_preds_tr[i])
    elif flag == 1:
        pred_tr.append(MLP_preds_tr[i])
    elif flag == 2:
        pred_tr.append(RF_preds_tr[i])
    elif flag == 3:
        pred_tr.append(MLP_preds_tr[i])


pred_te = []
for i, flag in enumerate(flags_YSO_te):
    if flag == 0:
        pred_te.append(MLP_preds_te[i])
    elif flag == 1:
        pred_te.append(MLP_preds_te[i])
    elif flag == 2:
        pred_te.append(RF_preds_te[i])
    elif flag == 3:
        pred_te.append(MLP_preds_te[i])

YSE_labels = ["YSO","EG","Stars"]
YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","EG","Stars"]

with open("../Results/"+outfile,"w") as f:
    f.write("MLP Results \n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,MLP_preds_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,MLP_preds_te,target_names=YSO_labels))
    f.write("\nCM Results\n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,RF_preds_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,RF_preds_te,target_names=YSO_labels))
    f.write("\nFlagging and with best classifications \n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,pred_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,pred_te,target_names=YSO_labels))