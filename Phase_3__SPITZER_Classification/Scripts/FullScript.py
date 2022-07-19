import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

import torch
import torch.utils.data as data_utils

from NN_Defs import BaseMLP, TwoLayerMLP, MLP_data_setup, test, preproc_yso
from custom_dataloader import replicate_data_single

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

# # Change preds to match up with new scheme
preproc_yso(inp_tr[:,-1],MLP_preds_tr)
preproc_yso(inp_te[:,-1],MLP_preds_te)

preproc_yso(inp_tr[:,-1],tar_tr)
preproc_yso(inp_te[:,-1],tar_te)

# Classify into YSO types
def to_loader(inp,tar):
    inp = torch.as_tensor(inp)
    tar = torch.as_tensor(tar)

    # pass tensors into TensorDataset instances
    tensor_data = data_utils.TensorDataset(inp, tar)

    # constructing data loaders
    loader = torch.utils.data.DataLoader(tensor_data, batch_size=32, shuffle=False)
    return loader

YSO_loader_tr = to_loader(inp_TR,tar_tr)
YSO_loader_te = to_loader(inp_TE,tar_te)
YSO_NN = BaseMLP(9, 20, 5)
YSO_NN.load_state_dict(torch.load("../MLP_Settings/IRAC_YSO_OneLayer_LR_0.1_MO_0.9_NEUR_20_Settings", map_location=device))

YSO_preds_tr = test(YSO_NN, YSO_loader_tr, device)
YSO_preds_te = test(YSO_NN, YSO_loader_te, device)

def flag_YSO(pred1,pred2):
    flag = []
    for i, p2 in enumerate(pred2):
        if pred1[i]==p2:
            flag.append(0)
        else:
            flag.append(1)
    return flag

flags_YSO_tr = np.array(flag_YSO(MLP_preds_tr,YSO_preds_tr))
flags_YSO_te = np.array(flag_YSO(MLP_preds_te,YSO_preds_te))

pred_tr = np.array(MLP_preds_tr)
# for i, flag in enumerate(flags_YSO_tr):
#     if flag == 0:
#         pred_tr.append(MLP_preds_tr[i])
#     elif flag == 1:
#         pred_tr.append(MLP_preds_tr[i])
#     elif flag == 2:
#         pred_tr.append(RF_preds_tr[i])
#     elif flag == 3:
#         pred_tr.append(MLP_preds_tr[i])

pred_te = np.array(MLP_preds_te)
# for i, flag in enumerate(flags_YSO_te):
#     if flag == 0:
#         pred_te.append(MLP_preds_te[i])
#     elif flag == 1:
#         pred_te.append(MLP_preds_te[i])
#     elif flag == 2:
#         pred_te.append(RF_preds_te[i])
#     elif flag == 3:
#         pred_te.append(MLP_preds_te[i])


YSE_labels = ["YSO","EG","Stars"]
YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","EG","Stars"]

with open("../Results/"+outfile,"w") as f:
    f.write("MLP on YSE Results \n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,MLP_preds_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,MLP_preds_te,target_names=YSO_labels))
    f.write("\nMLP on YSO Results\n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,YSO_preds_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,YSO_preds_te,target_names=YSO_labels))
    f.write("\nFlagging and with best classifications \n Training data (c2d Survey)\n")
    f.write(classification_report(tar_tr,pred_tr,target_names=YSO_labels))
    f.write("Testing data (NGC 2264)\n")
    f.write(classification_report(tar_te,pred_te,target_names=YSO_labels))


# t-SNE
def tsne_plot(inp,pred,flag,type):
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
    plt.scatter(Y[np.where(pred==4)[0], 0], Y[np.where(pred==4)[0], 1], c="orange",label='Stars')
    plt.scatter(Y[np.where(pred==3)[0], 0], Y[np.where(pred==3)[0], 1], c="purple",label='EG')
    plt.scatter(Y[np.where(pred==2)[0], 0], Y[np.where(pred==2)[0], 1], c="r",label='YSO - Class II')
    plt.scatter(Y[np.where(pred==1)[0], 0], Y[np.where(pred==1)[0], 1], c="b",label='YSO - Class FS')
    plt.scatter(Y[np.where(pred==0)[0], 0], Y[np.where(pred==0)[0], 1], c="g",label='YSO - Class I')
    plt.scatter(Y[np.where(flag==3)[0], 0], Y[np.where(flag==3)[0], 1], c="k",marker='x',label='Insecure')
    plt.legend()
    plt.savefig(f"../Results/t-SNE_final_{type}.png",dpi=300)

tsne_plot(inp_TR,pred_tr,flags_YSO_tr,"c2d")
tsne_plot(inp_TE,pred_te,flags_YSO_te,"NGC 2264")