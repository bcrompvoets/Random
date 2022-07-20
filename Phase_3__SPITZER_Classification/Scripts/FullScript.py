import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

import torch
import torch.utils.data as data_utils

from NN_Defs import BaseMLP, TwoLayerMLP, MLP_data_setup, test, preproc_yso
from custom_dataloader import replicate_data_single

device = torch.device("cpu")
outfile = "../Results/FullScript_Classification_Report_CIII.txt"
ClassIII = True
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
rfcl = RandomForestClassifier(class_weight='balanced',criterion='entropy',max_features='log2',n_estimators=50,oob_score=False)
rfcl.fit(inp_TR,tar_tr.ravel())
RF_preds_tr = rfcl.predict(inp_TR)
RF_preds_te = rfcl.predict(inp_TE)

# # Change preds to match up with new scheme
preproc_yso(inp_tr[:,-1],MLP_preds_tr,three=ClassIII)
preproc_yso(inp_te[:,-1],MLP_preds_te,three=ClassIII)

preproc_yso(inp_tr[:,-1],RF_preds_tr,three=ClassIII)
preproc_yso(inp_te[:,-1],RF_preds_te,three=ClassIII)

preproc_yso(inp_tr[:,-1],tar_tr,three=ClassIII)
preproc_yso(inp_te[:,-1],tar_te,three=ClassIII)
print(np.unique(tar_tr))
# Classify into YSO types
X_tr = np.load("../Data/c2d_YSO_INP.npy") # Load input data
Y_tr = np.load("../Data/c2d_YSO_TAR.npy") # Load target data
X_tr = np.float32(X_tr)
Y_tr = np.float32(Y_tr)
inp_tr_YSO, tar_tr_YSO = replicate_data_single(X_tr, Y_tr, [len(np.where(Y_tr==0.)[0])]*5)
inp_tr_YSO = np.delete(inp_tr_YSO,np.s_[8:10],axis=1)
YSO_forscale, YSO_train, YSO_test = MLP_data_setup(inp_tr_YSO, tar_tr_YSO,inp_TR, tar_tr, inp_TE, tar_te)

if ClassIII:
    YSO_NN = BaseMLP(9, 10, 6)
    YSO_NN.load_state_dict(torch.load("../MLP_Settings//IRAC_YSO_CIII_OneLayer_LR_0.1_MO_0.6_NEUR_10_Settings", map_location=device))
else:
    YSO_NN = BaseMLP(9, 20, 5)
    YSO_NN.load_state_dict(torch.load("../MLP_Settings/IRAC_YSO_OneLayer_LR_0.1_MO_0.9_NEUR_20_Settings", map_location=device))

YSO_preds_tr = test(YSO_NN, YSO_train, device)
YSO_preds_te = test(YSO_NN, YSO_test, device)

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
pred_tr = np.array(pred_tr)
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
pred_te = np.array(pred_te)


YSE_labels = ["YSO","EG","Stars"]
if ClassIII:
    YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","YSO - Class III","EG","Stars"]
else:    
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
    f.write("\nCM Results\n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,YSO_preds_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,YSO_preds_te,target_names=YSO_labels))
    f.write("\nFlagging and with best classifications \n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,pred_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,pred_te,target_names=YSO_labels))


# t-SNE
def tsne_plot(inp,pred,flag,type,three=False):
    n_components = 2
    tsne = TSNE(
        n_components=n_components,
        init="random",
        random_state=0,
        perplexity=30,
        learning_rate="auto",
        n_iter=300)
    Y = tsne.fit_transform(inp)

    plt.rcParams["font.family"] = "times"
    plt.title("t-SNE of predictions with insecure objects marked")
    if three:
        plt.scatter(Y[np.where(pred==5)[0], 0], Y[np.where(pred==5)[0], 1], c="orange",label='Stars')
        plt.scatter(Y[np.where(pred==4)[0], 0], Y[np.where(pred==4)[0], 1], c="purple",label='EG')
        plt.scatter(Y[np.where(pred==3)[0], 0], Y[np.where(pred==3)[0], 1], c="cyan",label='YSO - Class III')
    else:
        plt.scatter(Y[np.where(pred==4)[0], 0], Y[np.where(pred==4)[0], 1], c="orange",label='Stars')
        plt.scatter(Y[np.where(pred==3)[0], 0], Y[np.where(pred==3)[0], 1], c="purple",label='EG')
    plt.scatter(Y[np.where(pred==2)[0], 0], Y[np.where(pred==2)[0], 1], c="r",label='YSO - Class II')
    plt.scatter(Y[np.where(pred==1)[0], 0], Y[np.where(pred==1)[0], 1], c="b",label='YSO - Class FS')
    plt.scatter(Y[np.where(pred==0)[0], 0], Y[np.where(pred==0)[0], 1], c="g",label='YSO - Class I')
    plt.scatter(Y[np.where(flag==3)[0], 0], Y[np.where(flag==3)[0], 1], c="k",marker='x',label='Very Insecure')
    plt.scatter(Y[np.where(flag==2)[0], 0], Y[np.where(flag==2)[0], 1], c="w",marker='x',label='Insecure')
    plt.legend()
    plt.savefig(f"../Results/Figures/t-SNE_final_{type}.png",dpi=300)
    plt.close()

tsne_plot(inp_TR,pred_tr,flags_YSO_tr,"c2d_CIII",three=ClassIII)
tsne_plot(inp_TE,pred_te,flags_YSO_te,"NGC 2264_CIII",three=ClassIII)