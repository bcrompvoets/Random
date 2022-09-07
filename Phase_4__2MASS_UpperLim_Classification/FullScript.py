import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

import torch
import torch.utils.data as data_utils

from NN_Defs import BaseMLP, TwoLayerMLP, MLP_data_setup, test, preproc_yso
from custom_dataloader import replicate_data_single, replicate_data
from FullScript_Defs import predbyflag, tsne_plot, plot_hist, flag_ALL

device = torch.device("cpu")

# Define global variables
ClassIII = True
testset = "NGC 2264"#"c2d Full"
outfile = f"../Results/FullScript_Classification_Report_{testset}.txt"


# File to use to scale rest of data
file_tr = "c2d_w_quality.csv"
inputs = pd.read_csv(file_tr)
# Collect the column names of magnitudes and errors
bands = [idx for idx in inputs.columns.values if (idx[0].lower() == 'm'.lower() or idx[0].lower() == 'e'.lower())]
bands = bands[:-2] # Remove MIPS2
bands.append("alpha")
# print(bands)

inp_tr, tar_tr,inp_va, tar_va,inp_te, tar_te = replicate_data(inputs[bands].values.astype(float), inputs[['Target']].values.astype(int),[3000]*3,[1000,1650,10000])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])
while np.all(np.isfinite(inp_tr)) == False:
    inp_tr, tar_tr, inp_te, tar_te, inp_va, tar_va = replicate_data(inputs[bands].values.astype(float), inputs[['Target']].values.astype(int),[3000]*3,[1000,1650,10000])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])



# Add in test set here


# Test IRAC only
band = [idx for idx in inputs.columns.values if idx[-2].lower() == 'R'.lower()]
band_ind = np.where(np.isin(bands,band))[0]
IR_train, IR_valid, IR_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
NN_IR = TwoLayerMLP(len(band_ind), 10, 3)
NN_IR.load_state_dict(torch.load("Results/Best_Results/c2d_quality_2_IRAC_only/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
# Test MLP
IR_preds_tr = test(NN_IR, IR_train, device)
IR_preds_te = test(NN_IR, IR_test, device)

# Test IRAC and MIPS
band = [idx for idx in inputs.columns.values if (idx[-2].lower() == 'R'.lower() or idx[-2].lower() == 'P'.lower())]
band_ind = np.where(np.isin(bands,band))[0]
IM_train, IM_valid, IM_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
NN_IM = TwoLayerMLP(len(band_ind), 10, 3)
NN_IM.load_state_dict(torch.load("Results/Best_Results/c2d_quality_1/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
# Test MLP
IM_preds_tr = test(NN_IM, IM_train, device)
IM_preds_te = test(NN_IM, IM_test, device)

# Test IRAC and 2MASS
band = [idx for idx in inputs.columns.values if (idx[-2].lower() != 'P'.lower() and idx[0].lower() != 'a'.lower())]
band_ind = np.where(np.isin(bands,band))[0]
I2_train, I2_valid, I2_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
NN_I2 = TwoLayerMLP(len(band_ind), 20, 3)
NN_I2.load_state_dict(torch.load("Results/Best_Results/c2d_quality_3_IRAC_2MASS/TwoLayer_LR_0.001_MO__NEUR_20_Settings", map_location=device))
# Test MLP
I2_preds_tr = test(NN_I2, I2_train, device)
I2_preds_te = test(NN_I2, I2_test, device)

# Test IRAC, MIPS, and 2MASS
band = bands[:-1] # Everything except alpha
band_ind = np.where(np.isin(bands,band))[0]
I2M_train, I2M_valid, I2M_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
NN_I2M = TwoLayerMLP(len(band_ind), 20, 3)
NN_I2M.load_state_dict(torch.load("Results/Best_Results/c2d_quality_4_IRAC_MIPS_2MASS/TwoLayer_LR_0.001_MO__NEUR_20_Settings", map_location=device))
# Test MLP
I2M_preds_tr = test(NN_I2M, I2M_train, device)
I2M_preds_te = test(NN_I2M, I2M_test, device)


# Determine matching predictions by mode
# Combine predictions into nxm grid. n - number of objects, m - number of classifications used
Preds_tr = np.c_[IR_preds_tr,IM_preds_tr,I2_preds_tr,I2M_preds_tr]
PREDS_tr = stats.mode(Preds_tr.transpose())[0].ravel()
print(PREDS_tr)

tar_tr = np.hstack(tar_tr)

YSE_labels = ["YSO","EG","Stars"]
print(classification_report(tar_tr,PREDS_tr,target_names=YSE_labels))

# Preproc? Add in any sort of "most reliable" caveat instead of just the mode?
alph_ind = np.where(bands=='alpha')[0]
preproc_yso(inp_tr[:,-1],PREDS_tr,three=ClassIII)
preproc_yso(inp_tr[:,-1],tar_tr,three=ClassIII)

if ClassIII:
    YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","YSO - Class III","EG","Stars"]
else:    
    YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","EG","Stars"]

print("MLP Results \n Training data (c2d Survey)\n")
print(classification_report(tar_tr,PREDS_tr,target_names=YSO_labels))






# # YSO_EG_Stars Test
# X_te = np.load("../Data/NGC2264_INP.npy") # Load input data
# Y_te = np.load("../Data/NGC2264_TAR.npy") # Load target data
# X_te = np.float32(X_te)
# Y_te = np.float32(Y_te)
# X_te = np.c_[ 0:len(X_te), X_te ] # Add an index for now
# inp_te, tar_te = replicate_data_single(X_te, Y_te, [len(np.where(Y_te==0.)[0]),len(np.where(Y_te==1.)[0]),len(np.where(Y_te==2.)[0])])
# ind_te = inp_te[:,0] # Create array which only holds indices
# inp_te = inp_te[:,1:] # Remove index from input
# isf_te = np.all(np.isfinite(inp_te))
# if isf_te == False:
#     index = []
#     for i, j in enumerate(np.isfinite(inp_te)):
#         if np.all(j):
#             index.append(i)
#     inp_te = inp_te[index]
#     tar_te = tar_te[index]


# # REMOVE MIPS DATA FOR IRAC ONLY
# inp_TR = np.delete(inp_tr,np.s_[8:10],axis=1)
# inp_TE = np.delete(inp_te,np.s_[8:10],axis=1)

# IR_train, IR_valid, IR_test = MLP_data_setup(inp_TR, tar_tr,inp_TE, tar_te, inp_TE, tar_te)


# # IRAC MLP 1: Best MLP, trained on 10k objects in each class
# # Define and load in settings
# NN_IR = TwoLayerMLP(9, 20, 3)
# NN_IR.load_state_dict(torch.load("../MLP_Settings/IRAC_TwoLayer_LR_0.001_MO_0.9_NEUR_20_Settings", map_location=device))
# # Test MLP
# MLP_preds_tr = test(NN_IR, IR_train, device)
# MLP_preds_te = test(NN_IR, IR_test, device)

# YSE_labels = ["YSO","EG","Stars"]
# print(classification_report(tar_te,MLP_preds_te,target_names=YSE_labels))

# # IRAC MLP 2: Also very good, trained on 15k objects in each class (some synth)
# # Define and load in settings
# NN_IR_2 = BaseMLP(9, 10, 3)
# NN_IR_2.load_state_dict(torch.load("../MLP_Settings/IRAC_15k_OneLayer_LR_0.01_MO_0.75_NEUR_10_Settings", map_location=device))
# # Test MLP
# MLP_preds_tr_2 = test(NN_IR_2, IR_train, device)
# MLP_preds_te_2 = test(NN_IR_2, IR_test, device)

# # IRAC+MIPS MLP: Very good
# # Adjust test data to view only objects with MIPS detections
# mips_ind_tr = np.where(inp_tr[:,9]!=-99)[0]
# mips_ind_te = np.where(inp_te[:,9]!=-99)[0]
# inp_tr_M = inp_tr[mips_ind_tr]
# tar_tr_M = tar_tr[mips_ind_tr]
# inp_te_M = inp_te[mips_ind_te]
# tar_te_M = tar_te[mips_ind_te]
# train_M, valid_M, test_M = MLP_data_setup(inp_tr_M, tar_tr_M, inp_te_M, tar_te_M, inp_te_M, tar_te_M)
# # Define and load in settings
# MIPS_NN = BaseMLP(11,50,3)
# MIPS_NN.load_state_dict(torch.load("../MLP_Settings/MIPS_OneLayer_LR_0.1_MO_0.75_NEUR_50_Settings"))

# # Test MLP
# MLP_preds_tr_M = test(MIPS_NN,train_M,device)
# MLP_preds_te_M = test(MIPS_NN,test_M,device)

# # # Change preds to match up with new scheme
# preproc_yso(inp_tr[:,-1],MLP_preds_tr,three=ClassIII)
# preproc_yso(inp_te[:,-1],MLP_preds_te,three=ClassIII)

# preproc_yso(inp_tr[:,-1],MLP_preds_tr_2,three=ClassIII)
# preproc_yso(inp_te[:,-1],MLP_preds_te_2,three=ClassIII)

# preproc_yso(inp_tr[:,-1],tar_tr,three=ClassIII)
# preproc_yso(inp_te[:,-1],tar_te,three=ClassIII)

# preproc_yso(inp_tr_M[:,-1],MLP_preds_tr_M,three=ClassIII)
# preproc_yso(inp_te_M[:,-1],MLP_preds_te_M,three=ClassIII)
# preproc_yso(inp_tr_M[:,-1],tar_tr_M,three=ClassIII)
# preproc_yso(inp_te_M[:,-1],tar_te_M,three=ClassIII)

# # Classify into YSO types
# # Load in data for scaling according to the MLP trained on YSO classes
# X_tr = np.load("../Data/c2d_YSO_INP.npy") # Load input data
# Y_tr = np.load("../Data/c2d_YSO_TAR.npy") # Load target data
# X_tr = np.float32(X_tr)
# Y_tr = np.float32(Y_tr)
# inp_tr_YSO, tar_tr_YSO = replicate_data_single(X_tr, Y_tr, [len(np.where(Y_tr==0.)[0])]*5)
# inp_tr_YSO = np.delete(inp_tr_YSO,np.s_[8:10],axis=1)
# YSO_scale, YSO_train, YSO_test = MLP_data_setup(inp_tr_YSO, tar_tr_YSO,inp_TR, tar_tr, inp_TE, tar_te)

# # Choose which MLP to load in based on whether or not we are including YSO - Class III
# if ClassIII:
#     YSO_NN = BaseMLP(9, 10, 6)
#     YSO_NN.load_state_dict(torch.load("../MLP_Settings/IRAC_YSO_CIII_OneLayer_LR_0.1_MO_0.6_NEUR_10_Settings", map_location=device))
# else:
#     YSO_NN = BaseMLP(9, 20, 5)
#     YSO_NN.load_state_dict(torch.load("../MLP_Settings/IRAC_YSO_OneLayer_LR_0.1_MO_0.9_NEUR_20_Settings", map_location=device))

# YSO_preds_tr = test(YSO_NN, YSO_train, device)
# YSO_preds_te = test(YSO_NN, YSO_test, device)
# print(len(mips_ind_tr))
# # Flag predictions depending on security
# flags_YSO_tr = np.array(flag_ALL(MLP_preds_tr,YSO_preds_tr,MLP_preds_tr_2,MLP_preds_tr_M,mips_ind_tr))
# flags_YSO_te = np.array(flag_ALL(MLP_preds_te,YSO_preds_te,MLP_preds_te_2,MLP_preds_te_M,mips_ind_te))
# print(len(flags_YSO_tr))
# print(len(MLP_preds_tr))
# print(len(MLP_preds_tr_2))
# # Define predictions based off of flags
# pred_tr = np.array(predbyflag(flags_YSO_tr,inp_tr[:,-1],MLP_pred=MLP_preds_tr,MLP_pred_2=MLP_preds_tr_2,YSO_pred=YSO_preds_tr,MLP_pred_M=MLP_preds_tr_M,mips_ind=mips_ind_tr,ClassIII=ClassIII))
# pred_te = np.array(predbyflag(flags_YSO_te,inp_te[:,-1],MLP_pred=MLP_preds_te,MLP_pred_2=MLP_preds_te_2,YSO_pred=YSO_preds_te,MLP_pred_M=MLP_preds_te_M,mips_ind=mips_ind_te,ClassIII=ClassIII))
# pred_tr = np.vstack(pred_tr)
# pred_te = np.vstack(pred_te)

# print("Predictions completed")
# print(f"We have {len(np.where(pred_te==0)[0])} Class I YSOs")
# print(f"We have {len(np.where(pred_te==1)[0])} Class FS YSOs")
# print(f"We have {len(np.where(pred_te==2)[0])} Class II YSOs")
# print(f"We have {len(np.where(pred_te==3)[0])} Class III YSOs")
# print(f"We have {len(np.where(pred_te==4)[0])} EGs")
# print(f"We have {len(np.where(pred_te==5)[0])} Stars")
# print(f"We have {len(np.where(flags_YSO_te==3)[0])} Insecure Classifications")









# # t-SNE
# tsne_plot(inp_TR,pred_tr,flags_YSO_tr,"c2d_CIII_2YSE_MIPS",three=ClassIII)
# tsne_plot(inp_TE,pred_te,flags_YSO_te,f"{testset}_CIII_2YSE_MIPS",three=ClassIII)
# print("t-SNE plots completed")

# # Histogram
# plot_hist(inp_TR,pred_tr,"c2d_CIII_2YSE_MIPS",ClassIII=ClassIII)
# plot_hist(inp_TE,pred_te,f"{testset}_CIII_2YSE_MIPS",ClassIII=ClassIII)
# print("Histograms of spectral index completed")

# # Print the Classification Reports
# YSE_labels = ["YSO","EG","Stars"]
# if ClassIII:
#     YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","YSO - Class III","EG","Stars"]
# else:    
#     YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","EG","Stars"]

# with open("../Results/"+outfile,"w") as f:
#     f.write("MLP Results \n Training data (c2d Survey)\n")
#     f.write(classification_report(tar_tr,MLP_preds_tr,target_names=YSO_labels))
#     f.write(f"Testing data ({testset})\n")
#     f.write(classification_report(tar_te,MLP_preds_te,target_names=YSO_labels))
#     f.write("\nMLP from YSO Results\n Training data (c2d Survey)\n")
#     f.write(classification_report(tar_tr,YSO_preds_tr,target_names=YSO_labels))
#     f.write(f"Testing data ({testset})\n")
#     f.write(classification_report(tar_te,YSO_preds_te,target_names=YSO_labels))
#     f.write("\nMLP from YSE 2 Results\n Training data (c2d Survey)\n")
#     f.write(classification_report(tar_tr,MLP_preds_tr_2,target_names=YSO_labels))
#     f.write(f"Testing data ({testset})\n")
#     f.write(classification_report(tar_te,MLP_preds_te_2,target_names=YSO_labels))
#     f.write("\nMLP from YSE MIPS Results\n Training data (c2d Survey)\n")
#     f.write(classification_report(tar_tr_M,MLP_preds_tr_M,target_names=YSO_labels))
#     f.write(f"Testing data ({testset})\n")
#     f.write(classification_report(tar_te_M,MLP_preds_te_M,target_names=YSO_labels))
#     f.write("\nFlagging and with best classifications \n Training data (c2d Survey)\n")
#     f.write(classification_report(tar_tr,pred_tr,target_names=YSO_labels))
#     f.write(f"Testing data ({testset})\n")
#     f.write(classification_report(tar_te,pred_te,target_names=YSO_labels))
#     # Make comparison of og three classes:
#     YSE_tar_te = tar_te
#     if ClassIII:
#         YSE_tar_te[YSE_tar_te<=3]=0
#         YSE_tar_te[YSE_tar_te==4]=1
#         YSE_tar_te[YSE_tar_te==5]=2

#         YSE_pred_te = pred_te
#         YSE_pred_te[YSE_pred_te<=3]=0
#         YSE_pred_te[YSE_pred_te==4]=1
#         YSE_pred_te[YSE_pred_te==5]=2
#     else:
#         YSE_tar_te[YSE_tar_te<=2]=0
#         YSE_tar_te[YSE_tar_te==3]=1
#         YSE_tar_te[YSE_tar_te==4]=2

#         YSE_pred_te = pred_te
#         YSE_pred_te[YSE_pred_te<=2]=0
#         YSE_pred_te[YSE_pred_te==3]=1
#         YSE_pred_te[YSE_pred_te==4]=2
#     f.write(f"Testing data ({testset})\n")
#     f.write(classification_report(YSE_tar_te,pred_te,target_names=YSE_labels))

# bins = np.linspace(-6,6,60)
# ind_Y = np.where(YSE_pred_te==0)[0]
# ind_E = np.where(YSE_pred_te==1)[0]
# ind_S = np.where(YSE_pred_te==2)[0]
# plt.hist(inp_te[ind_Y,-1],bins,histtype='step',density=True, color="mediumspringgreen", label = "YSOs")
# plt.hist(inp_te[ind_E,-1],bins,histtype='step',density=True, color="gold", label= "EGs")
# plt.hist(inp_te[ind_S,-1],bins,histtype='step',density=True, color="green", label = "Stars")

# # binwidth=0.2
# mu_Y = np.mean(inp_te[ind_Y,-1])
# sig_Y = np.std(inp_te[ind_Y,-1])
# # yM_Y = binwidth*len(ind_Y)
# mu_E = np.mean(inp_te[ind_E,-1])
# sig_E = np.std(inp_te[ind_E,-1])
# # yM_E = binwidth*len(ind_E)
# mu_S = np.mean(inp_te[ind_S,-1])
# sig_S = np.std(inp_te[ind_S,-1])
# # yM_S = binwidth*len(ind_S)
# bins_gaus = np.linspace(-6,6,600)
# plt.plot(bins_gaus, (1/(sig_Y * np.sqrt(2 * np.pi)) * np.exp( - (bins_gaus - mu_Y)**2 / (2 * sig_Y**2)) ),linewidth=2, color='mediumspringgreen')
# plt.plot(bins_gaus, (1/(sig_E * np.sqrt(2 * np.pi)) * np.exp( - (bins_gaus - mu_E)**2 / (2 * sig_E**2)) ),linewidth=2, color='gold')
# plt.plot(bins_gaus, (1/(sig_S * np.sqrt(2 * np.pi)) * np.exp( - (bins_gaus - mu_S)**2 / (2 * sig_S**2)) ),linewidth=2, color='green')

# plt.axvline(-2,ymax=1,ymin=0,color='k')
# plt.title("Spectral indices of YSOs, EGs, and Stars")
# plt.xlabel("Spectral index α")
# plt.legend()
# plt.savefig(f"../Results/Figures/{testset}_YSE_Hist.png",dpi=300)