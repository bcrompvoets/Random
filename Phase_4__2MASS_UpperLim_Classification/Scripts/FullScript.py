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
testset = "NGC_2264"#"c2d Full"
outfile_tr = f"../Results/Classification_Reports/c2d.txt"
outfile_te = f"../Results/Classification_Reports/{testset}.txt"
f_tr = open(outfile_tr,'w')
f_te = open(outfile_te,'w')

# File to use to scale rest of data
file_tr = "c2d_w_quality.csv"
inputs = pd.read_csv(file_tr)
# Collect the column names of magnitudes and errors
bands = [idx for idx in inputs.columns.values if (idx[0].lower() == 'm'.lower() or idx[0].lower() == 'e'.lower())]
bands = bands[:-2] # Remove MIPS2
bands.append("alpha")
# print(bands)


# Add in test set here
testin = pd.read_csv(f"{testset}.csv")
# Collect the column names of magnitudes and errors
bands = [idx for idx in testin.columns.values if (idx[0].lower() == 'm'.lower() or idx[0].lower() == 'e'.lower())]
bands.append("alpha")

inp_tr, tar_tr = replicate_data_single(inputs[bands].values.astype(float), inputs[['Target']].values.astype(int),[len(np.where(inputs[['Target']].values==0)[0])]*3)#,len(np.where(inputs[['Target']].values==1)[0]),int(len(np.where(inputs[['Target']].values==2)[0])/100)])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])
while np.all(np.isfinite(inp_tr)) == False:
    inp_tr, tar_tr = replicate_data_single(inputs[bands].values.astype(float), inputs[['Target']].values.astype(int),[len(np.where(inputs[['Target']].values==0)[0])]*3)#,len(np.where(inputs[['Target']].values==1)[0]),int(len(np.where(inputs[['Target']].values==2)[0])/100)])


inp_te, tar_te = replicate_data_single(testin[bands].values.astype(float), testin[['Target']].values.astype(int),[len(np.where(testin[['Target']].values==0)[0]),len(np.where(testin[['Target']].values==1)[0]),len(np.where(testin[['Target']].values==2)[0])])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])
while np.all(np.isfinite(inp_te)) == False:
    inp_te, tar_te = replicate_data_single(testin[bands].values.astype(float), testin[['Target']].values.astype(int),[len(np.where(testin[['Target']].values==0)[0]),len(np.where(testin[['Target']].values==1)[0]),len(np.where(testin[['Target']].values==2)[0])])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])


# Add in only networks for which you have data:
if bands[0]=='mag_J':
    bands[0] = 'mag_2M1'
b = [idx[slice(-3,-1)] for idx in bands if (idx[-1] == '1' and idx[0] != 'e')]
print(b)
f_tr.write(f"Using only {b} bands\n")
f_te.write(f"Using only {b} bands\n")
# Test IRAC only
band = [idx for idx in bands if idx[-2].lower() == 'R'.lower()]
band_ind = np.where(np.isin(bands,band))[0]
IR_train, IR_valid, IR_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
NN_IR = TwoLayerMLP(len(band_ind), 10, 3)
NN_IR.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_2_IRAC_only/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
# Test MLP
IR_preds_tr = test(NN_IR, IR_train, device)
IR_preds_te = test(NN_IR, IR_test, device)
if "MP" in b[-1] and "2M" in b[0]:
    # Test IRAC, MIPS, and 2MASS
    band = bands[:-1] # Everything except alpha
    band_ind = np.where(np.isin(bands,band))[0]
    I2M_train, I2M_valid, I2M_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
    NN_I2M = TwoLayerMLP(len(band_ind), 20, 3)
    NN_I2M.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_4_IRAC_MIPS_2MASS/TwoLayer_LR_0.001_MO__NEUR_20_Settings", map_location=device))
    # Test MLP
    I2M_preds_tr = test(NN_I2M, I2M_train, device)
    I2M_preds_te = test(NN_I2M, I2M_test, device)
else:
    I2M_preds_tr = [np.nan]*len(IR_preds_tr)
    I2M_preds_te = [np.nan]*len(IR_preds_te)
if "MP" in b[-1]:
    # Test IRAC and MIPS
    band = [idx for idx in bands if (idx[-2].lower() == 'R'.lower() or idx[-2].lower() == 'P'.lower())]
    band_ind = np.where(np.isin(bands,band))[0]
    IM_train, IM_valid, IM_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
    NN_IM = TwoLayerMLP(len(band_ind), 10, 3)
    NN_IM.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_1/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
    # Test MLP
    IM_preds_tr = test(NN_IM, IM_train, device)
    IM_preds_te = test(NN_IM, IM_test, device)
else:
    IM_preds_tr = [np.nan]*len(IR_preds_tr)
    IM_preds_te = [np.nan]*len(IR_preds_te)
if "2M" in b[0]:
    # Test IRAC and 2MASS
    band = [idx for idx in bands if (idx[-2].lower() != 'P'.lower() and idx.lower()!='alpha')]
    band_ind = np.where(np.isin(bands,band))[0]
    I2_train, I2_valid, I2_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
    NN_I2 = TwoLayerMLP(len(band_ind), 20, 3)
    NN_I2.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_3_IRAC_2MASS/TwoLayer_LR_0.001_MO__NEUR_20_Settings", map_location=device))
    # Test MLP
    I2_preds_tr = test(NN_I2, I2_train, device)
    I2_preds_te = test(NN_I2, I2_test, device)
else:
    I2_preds_tr = [np.nan]*len(IR_preds_tr)
    I2_preds_te = [np.nan]*len(IR_preds_te)



# Determine matching predictions by mode
# Combine predictions into nxm grid. n - number of objects, m - number of classifications used
PREDS_tr = stats.mode(np.c_[IR_preds_tr,IM_preds_tr,I2_preds_tr,I2M_preds_tr].transpose(),nan_policy='omit')[0].ravel()
PREDS_te = stats.mode(np.c_[IR_preds_te,IM_preds_te,I2_preds_te,I2M_preds_te].transpose(),nan_policy='omit')[0].ravel()

tar_tr = np.hstack(tar_tr)

YSE_labels = ["YSO","EG","Stars"]

f_tr.write("YSE Results Training Set\n")
f_tr.write(classification_report(tar_tr,PREDS_tr,target_names=YSE_labels))
f_te.write(f"YSE Results Validation Set {testset}\n")
f_te.write(classification_report(tar_te,PREDS_te,target_names=YSE_labels))

# Preproc? Add in any sort of "most reliable" caveat instead of just the mode?
preproc_yso(inp_tr[:,-1],PREDS_tr,three=ClassIII)
preproc_yso(inp_tr[:,-1],tar_tr,three=ClassIII)
preproc_yso(inp_te[:,-1],PREDS_te,three=ClassIII)
preproc_yso(inp_te[:,-1],tar_te,three=ClassIII)


# Add in classification of YSOs from network
yso_ind_tr = np.where(PREDS_tr <= 3)[0]
yso_ind_te = np.where(PREDS_te <= 3)[0]
# Test IRAC only
band = [idx for idx in bands if idx[-2].lower() == 'R'.lower()]
band.append("alpha")
band_ind = np.where(np.isin(bands,band))[0]
IRY_train, IRY_valid, IRY_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], PREDS_tr[yso_ind_tr], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te])
NN_IRY = TwoLayerMLP(len(band_ind), 10, 4)
NN_IRY.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_8_IRAC_alpha_YSO/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
# Test MLP
IRY_preds_tr = test(NN_IRY, IRY_train, device)
IRY_preds_te = test(NN_IRY, IRY_test, device)
if "MP" in b[-1] and "2M" in b[0]:
    # Test IRAC, MIPS, and 2MASS
    band = bands
    band_ind = np.where(np.isin(bands,band))[0]
    I2MY_train, I2MY_valid, I2MY_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], PREDS_tr[yso_ind_tr], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te])
    NN_I2MY = TwoLayerMLP(len(band_ind), 10, 4)
    NN_I2MY.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_11_IRAC_MIPS_2MASS_alpha_YSO/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
    # Test MLP
    I2MY_preds_tr = test(NN_I2MY, I2MY_train, device)
    I2MY_preds_te = test(NN_I2MY, I2MY_test, device)
else:
    I2MY_preds_tr = [np.nan]*len(IRY_preds_tr)
    I2MY_preds_te = [np.nan]*len(IRY_preds_te)
if "MP" in b[-1]:
    # Test IRAC and MIPS
    band = [idx for idx in bands if (idx[-2].lower() == 'R'.lower() or idx[-2].lower() == 'P'.lower())]
    band.append("alpha")
    band_ind = np.where(np.isin(bands,band))[0]
    IMY_train, IMY_valid, IMY_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], PREDS_tr[yso_ind_tr], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te])
    NN_IMY = TwoLayerMLP(len(band_ind), 10, 4)
    NN_IMY.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_9_IRAC_MIPS_alpha_YSO/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
    # Test MLP
    IMY_preds_tr = test(NN_IMY, IMY_train, device)
    IMY_preds_te = test(NN_IMY, IMY_test, device)
else:
    IMY_preds_tr = [np.nan]*len(IRY_preds_tr)
    IMY_preds_te = [np.nan]*len(IRY_preds_te)
if "2M" in b[0]:
    # Test IRAC and 2MASS
    band = [idx for idx in bands if (idx[-2].lower() != 'P'.lower())]
    band_ind = np.where(np.isin(bands,band))[0]
    I2Y_train, I2Y_valid, I2Y_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], PREDS_tr[yso_ind_tr], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te], inp_te[yso_ind_te,:][:,band_ind], PREDS_te[yso_ind_te])
    NN_I2Y = TwoLayerMLP(len(band_ind), 20, 4)
    NN_I2Y.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_10_IRAC_2MASS_alpha_YSO/TwoLayer_LR_0.001_MO__NEUR_20_Settings", map_location=device))
    # Test MLP
    I2Y_preds_tr = test(NN_I2Y, I2Y_train, device)
    I2Y_preds_te = test(NN_I2Y, I2Y_test, device)
else:
    I2Y_preds_tr = [np.nan]*len(IRY_preds_tr)
    I2Y_preds_te = [np.nan]*len(IRY_preds_te)
    

YSO_preds_tr = stats.mode(np.c_[IRY_preds_tr,IMY_preds_tr,I2Y_preds_tr,I2MY_preds_tr].transpose(),nan_policy='omit')[0].ravel()
PREDS_tr[yso_ind_tr]=YSO_preds_tr
YSO_preds_te = stats.mode(np.c_[IRY_preds_te,IMY_preds_te,I2Y_preds_te,I2MY_preds_te].transpose(),nan_policy='omit')[0].ravel()
PREDS_te[yso_ind_te]=YSO_preds_te

if ClassIII:
    YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","YSO - Class III","EG","Stars"]
else:    
    YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","EG","Stars"]

f_tr.write("YSO Results Training Set\n")
f_tr.write(classification_report(tar_tr,PREDS_tr,target_names=YSO_labels))
f_te.write(f"YSO Results Validation Set {testset}\n")
f_te.write(classification_report(tar_te,PREDS_te,target_names=YSO_labels))


f_tr.write(f"We have {len(np.where(PREDS_tr==0)[0])} Class I YSOs\n")
f_tr.write(f"We have {len(np.where(PREDS_tr==1)[0])} Class FS YSOs\n")
f_tr.write(f"We have {len(np.where(PREDS_tr==2)[0])} Class II YSOs\n")
f_tr.write(f"We have {len(np.where(PREDS_tr==3)[0])} Class III YSOs\n")
f_tr.write(f"We have {len(np.where(PREDS_tr==4)[0])} EGs\n")
f_tr.write(f"We have {len(np.where(PREDS_tr==5)[0])} Stars\n")

f_te.write(f"We have {len(np.where(PREDS_te==0)[0])} Class I YSOs\n")
f_te.write(f"We have {len(np.where(PREDS_te==1)[0])} Class FS YSOs\n")
f_te.write(f"We have {len(np.where(PREDS_te==2)[0])} Class II YSOs\n")
f_te.write(f"We have {len(np.where(PREDS_te==3)[0])} Class III YSOs\n")
f_te.write(f"We have {len(np.where(PREDS_te==4)[0])} EGs\n")
f_te.write(f"We have {len(np.where(PREDS_te==5)[0])} Stars\n")

f_tr.close()
f_te.close()

print("Predictions completed")
print(f"We have {len(np.where(PREDS_te==0)[0])} Class I YSOs")
print(f"We have {len(np.where(PREDS_te==1)[0])} Class FS YSOs")
print(f"We have {len(np.where(PREDS_te==2)[0])} Class II YSOs")
print(f"We have {len(np.where(PREDS_te==3)[0])} Class III YSOs")
print(f"We have {len(np.where(PREDS_te==4)[0])} EGs")
print(f"We have {len(np.where(PREDS_te==5)[0])} Stars")
# print(f"We have {len(np.where(flags_YSO_te==3)[0])} Insecure Classifications")


# t-SNE
# tsne_plot(inp_TR,pred_tr,flags_YSO_tr,"c2d_CIII_2YSE_MIPS",three=ClassIII)
# tsne_plot(inp_TE,pred_te,flags_YSO_te,f"{testset}_CIII_2YSE_MIPS",three=ClassIII)
# print("t-SNE plots completed")

# Histogram
plot_hist(inp_tr,PREDS_tr,"spectral_index_hist_c2d",ClassIII=ClassIII)
plot_hist(inp_te,PREDS_te,f"spectral_index_hist_{testset}",ClassIII=ClassIII)
print("Histograms of spectral index completed")

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