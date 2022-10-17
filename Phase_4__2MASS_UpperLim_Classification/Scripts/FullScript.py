from cProfile import label
from locale import normalize
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import random

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import torch
import torch.utils.data as data_utils

from NN_Defs import BaseMLP, TwoLayerMLP, MLP_data_setup, test, preproc_yso
from custom_dataloader import replicate_data_single, replicate_data
from FullScript_Defs import predict_yse, tsne_plot, plot_hist

device = torch.device("cpu")

# Define global variables
ClassIII = True
Predict = False
testdir = "SPICY_w_quality"
testset = testdir
outfile_tr = f"../Results/Classification_Reports/c2d.txt"
outfile_te = f"../Results/Classification_Reports/{testset}.txt"
f_tr = open(outfile_tr,'w')
f_te = open(outfile_te,'w')

# File to use to scale rest of data
file_tr = "c2d_to_scale.csv"
inputs = pd.read_csv(file_tr)
# Collect the column names of magnitudes and errors
# bands = [idx for idx in inputs.columns.values if (idx[0].lower() == 'm'.lower() or idx[0].lower() == 'e'.lower())]
# bands = bands[:-2] # Remove MIPS2
# bands.append("alpha")
# print(bands)


# Add in test set here
testin = pd.read_csv(f"{testdir}.csv",comment='#')
# Collect the column names of magnitudes and errors
bands = [idx for idx in testin.columns.values if (idx[0].lower() == 'm'.lower() or idx[0].lower() == 'e'.lower())]
bands.append("alpha")

bands_TR = [idx for idx in inputs.columns.values if (idx[0].lower() == 'm'.lower() or idx[0].lower() == 'e'.lower())]
bands_TR.append("alpha")

print(np.unique(testin['Target'].values))
testin = testin[testin.Target < 3]
print(np.unique(testin['Target'].values))
# testin['Target'] = np.random.randint(low = 0,high=3,size=testin.shape[0])
if 'Target' not in testin.columns.values:
    testin['Target'] = np.random.randint(low = 0,high=3,size=testin.shape[0])
    # testin['Target'] = [0]*len(testin[['alpha']].values)
Predict = True


inp_tr, tar_tr = replicate_data_single(inputs[bands].values.astype(float), inputs[['Target']].values.astype(int),[len(np.where(inputs[['Target']].values==0)[0])]*3)#,len(np.where(inputs[['Target']].values==1)[0]),int(len(np.where(inputs[['Target']].values==2)[0])/100)])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])
while np.all(np.isfinite(inp_tr)) == False:
    inp_tr, tar_tr = replicate_data_single(inputs[bands].values.astype(float), inputs[['Target']].values.astype(int),[len(np.where(inputs[['Target']].values==0)[0])]*3)#,len(np.where(inputs[['Target']].values==1)[0]),int(len(np.where(inputs[['Target']].values==2)[0])/100)])

inp_te, tar_te = replicate_data_single(testin[bands].values.astype(float), testin[['Target']].values.astype(int),[len(np.where(testin[['Target']].values==0)[0]),len(np.where(testin[['Target']].values==1)[0]),len(np.where(testin[['Target']].values==2)[0])])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])
while np.all(np.isfinite(inp_te)) == False:
    inp_te, tar_te = replicate_data_single(testin[bands].values.astype(float), testin[['Target']].values.astype(int),[len(np.where(testin[['Target']].values==0)[0]),len(np.where(testin[['Target']].values==1)[0]),len(np.where(testin[['Target']].values==2)[0])])#,len(np.where(Y_te==3.)[0]),len(np.where(Y_te==4.)[0]),len(np.where(Y_te==5.)[0])])

if bands[0]=='mag_J':
    bands[0] = 'mag_2M1'
b = [idx[slice(-3,-1)] for idx in bands if (idx[-1] == '1' and idx[0] != 'e')]
print(b)
f_tr.write(f"Using only {b} bands\n")
f_te.write(f"Using only {b} bands\n")

print(bands)
Preds_tr, Preds_te, flags_tr, flags_te = predict_yse(inp_tr, tar_tr, inp_te, tar_te, bands, device)


PREDS_tr = stats.mode(Preds_tr,axis=1,nan_policy='omit')[0].ravel().astype(int)
PREDS_te = stats.mode(Preds_te,axis=1,nan_policy='omit')[0].ravel().astype(int)

tar_tr = np.hstack(tar_tr)
tar_te = np.hstack(tar_te)
YSE_labels = ["YSO","EG","Stars"]

f_tr.write("YSE Results Training Set\n")
f_tr.write(classification_report(tar_tr,PREDS_tr,target_names=YSE_labels))
f_te.write(f"YSE Results Validation Set {testset}\n")
f_te.write(classification_report(tar_te,PREDS_te,target_names=YSE_labels))

# Preproc? Add in any sort of "most reliable" caveat instead of just the mode?
preproc_yso(inp_tr[:,-1],PREDS_tr,three=ClassIII)
preproc_yso(inp_tr[:,-1],tar_tr,three=ClassIII)
# preproc_yso(inp_te[:,-1],PREDS_te,three=ClassIII)
preproc_yso(inp_te[:,-1],tar_te,three=ClassIII)

yso_ind_tr = np.array(np.where(~Preds_tr.all(axis=1))[0]) # Since Preds_tr.all(axis=1) returns which indices only contain greater than 0, using the not operator (~) returns true false array of yso indices being true
yso_ind_te = np.array(np.where(~Preds_te.all(axis=1))[0])
for i in Preds_tr.transpose():
    i = preproc_yso(inp_tr[:,-1],i,three=ClassIII)
for i in Preds_te.transpose():
    i = preproc_yso(inp_te[:,-1],i,three=ClassIII)

filler_yso_tr = [0]*len(yso_ind_tr)
filler_yso_te = [0]*len(yso_ind_te)
# Add in classification of YSOs from network
# Test IRAC only
band = [idx for idx in bands if idx[-2].lower() == 'R'.lower()]
band.append("alpha")
band_ind = np.where(np.isin(bands,band))[0]
IRY_train, IRY_valid, IRY_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], filler_yso_tr, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te)
NN_IRY = TwoLayerMLP(len(band_ind), 20, 6)
NN_IRY.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_15_IRAC_alpha_YSO+SE/TwoLayer_LR_0.001_MO__NEUR_20_Settings", map_location=device))
# Test MLP
IRY_preds_tr = np.array([np.nan]*len(PREDS_tr))
IRY_preds_te = np.array([np.nan]*len(PREDS_te))
IRY_preds_tr[yso_ind_tr] = test(NN_IRY, IRY_train, device)
IRY_preds_te[yso_ind_te] = test(NN_IRY, IRY_test, device)


I2MY_preds_tr = np.array([np.nan]*len(PREDS_tr))
I2MY_preds_te = np.array([np.nan]*len(PREDS_te))
if "MP" in b[-1] and "2M" in b[0]:
    # Test IRAC, MIPS, and 2MASS
    band = bands
    band_ind = np.where(np.isin(bands,band))[0]
    I2MY_train, I2MY_valid, I2MY_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], filler_yso_tr, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te)
    NN_I2MY = TwoLayerMLP(len(band_ind), 10, 6)
    NN_I2MY.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_12_IRAC_MIPS_2MASS_alpha_YSO+SE/TwoLayer_LR_0.001_MO__NEUR_10_Settings", map_location=device))
    # Test MLP
    I2MY_preds_tr[yso_ind_tr] = test(NN_I2MY, I2MY_train, device)
    I2MY_preds_te[yso_ind_te] = test(NN_I2MY, I2MY_test, device)

IMY_preds_tr = np.array([np.nan]*len(PREDS_tr))
IMY_preds_te = np.array([np.nan]*len(PREDS_te))
if "MP" in b[-1]:
    # Test IRAC and MIPS
    band = [idx for idx in bands if (idx[-2].lower() == 'R'.lower() or idx[-2].lower() == 'P'.lower())]
    band.append("alpha")
    band_ind = np.where(np.isin(bands,band))[0]
    IMY_train, IMY_valid, IMY_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], filler_yso_tr, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te)
    NN_IMY = TwoLayerMLP(len(band_ind), 20, 6)
    NN_IMY.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_16_LR_Reduce_YSO+SE/IRAC_MIPS_alphaLR_0.001_MO__NEUR_20_Settings", map_location=device))
    # Test MLP
    IMY_preds_tr[yso_ind_tr] = test(NN_IMY, IMY_train, device)
    IMY_preds_te[yso_ind_te] = test(NN_IMY, IMY_test, device)


I2Y_preds_tr = np.array([np.nan]*len(PREDS_tr))
I2Y_preds_te = np.array([np.nan]*len(PREDS_te))
if "2M" in b[0]:
    # Test IRAC and 2MASS
    band = [idx for idx in bands if (idx[-2].lower() != 'P'.lower())]
    band_ind = np.where(np.isin(bands,band))[0]
    I2Y_train, I2Y_valid, I2Y_test = MLP_data_setup(inp_tr[yso_ind_tr,:][:,band_ind], filler_yso_tr, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te, inp_te[yso_ind_te,:][:,band_ind], filler_yso_te)
    NN_I2Y = TwoLayerMLP(len(band_ind), 20, 6)
    NN_I2Y.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_16_LR_Reduce_YSO+SE/IRAC_2MASS_alphaLR_0.001_MO__NEUR_20_Settings", map_location=device))
    # Test MLP
    I2Y_preds_tr[yso_ind_tr] = test(NN_I2Y, I2Y_train, device)
    I2Y_preds_te[yso_ind_te] = test(NN_I2Y, I2Y_test, device)
    

YSO_PREDS_tr = stats.mode(np.c_[Preds_tr,IRY_preds_tr,IMY_preds_tr,I2Y_preds_tr,I2MY_preds_tr],axis=1,nan_policy='omit')[0].ravel()

YSO_PREDS_te = stats.mode(np.c_[Preds_te,IRY_preds_te,IMY_preds_te,I2Y_preds_te,I2MY_preds_te],axis=1,nan_policy='omit')[0].ravel()


if ClassIII:
    YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","YSO - Class III","EG","Stars"]
else:    
    YSO_labels = ["YSO - Class I","YSO - Class FS","YSO - Class II","EG","Stars"]

f_tr.write("YSO Results Training Set\n")
f_tr.write(classification_report(tar_tr,YSO_PREDS_tr,target_names=YSO_labels))
f_te.write(f"YSO Results Validation Set {testset}\n")
f_te.write(classification_report(tar_te,YSO_PREDS_te,target_names=YSO_labels))


f_tr.write(f"We have {len(np.where(YSO_PREDS_tr==0)[0])} Class I YSOs\n")
f_tr.write(f"We have {len(np.where(YSO_PREDS_tr==1)[0])} Class FS YSOs\n")
f_tr.write(f"We have {len(np.where(YSO_PREDS_tr==2)[0])} Class II YSOs\n")
f_tr.write(f"We have {len(np.where(YSO_PREDS_tr==3)[0])} Class III YSOs\n")
f_tr.write(f"We have {len(np.where(YSO_PREDS_tr==4)[0])} EGs\n")
f_tr.write(f"We have {len(np.where(YSO_PREDS_tr==5)[0])} Stars\n")
f_tr.write(f"We have {len(np.where(flags_tr==1)[0])} Insecure Classifications")

f_te.write(f"We have {len(np.where(YSO_PREDS_te==0)[0])} Class I YSOs\n")
f_te.write(f"We have {len(np.where(YSO_PREDS_te==1)[0])} Class FS YSOs\n")
f_te.write(f"We have {len(np.where(YSO_PREDS_te==2)[0])} Class II YSOs\n")
f_te.write(f"We have {len(np.where(YSO_PREDS_te==3)[0])} Class III YSOs\n")
f_te.write(f"We have {len(np.where(YSO_PREDS_te==4)[0])} EGs\n")
f_te.write(f"We have {len(np.where(YSO_PREDS_te==5)[0])} Stars\n")
f_te.write(f"We have {len(np.where(flags_te==1)[0])} Insecure Classifications")

f_tr.close()
f_te.close()

print("Predictions completed")
print(f"We have {len(np.where(YSO_PREDS_te==0)[0])} Class I YSOs")
print(f"We have {len(np.where(YSO_PREDS_te==1)[0])} Class FS YSOs")
print(f"We have {len(np.where(YSO_PREDS_te==2)[0])} Class II YSOs")
print(f"We have {len(np.where(YSO_PREDS_te==3)[0])} Class III YSOs")
print(f"We have {len(np.where(YSO_PREDS_te==4)[0])} EGs")
print(f"We have {len(np.where(YSO_PREDS_te==5)[0])} Stars")
print(f"We have {len(np.where(flags_te==1)[0])} Insecure Classifications")


# t-SNE
tsne_plot(inp_tr,YSO_PREDS_tr,flags_tr,"c2d",three=ClassIII)
tsne_plot(inp_te,YSO_PREDS_te,flags_te,f"{testset}",three=ClassIII)
print("t-SNE plots completed")

# Histogram
plot_hist(inp_tr,YSO_PREDS_tr,"spectral_index_hist_c2d",ClassIII=ClassIII)
plot_hist(inp_te,YSO_PREDS_te,f"spectral_index_hist_{testset}",ClassIII=ClassIII)
print("Histograms of spectral index completed")



if Predict:
    df = pd.DataFrame(data=np.c_[inp_te,PREDS_te],columns=np.r_[bands,['Preds']])
    df.to_csv(f'{testset}_w_Preds.csv')


# Confusion Matrix
plt.subplot()
ConfusionMatrixDisplay.from_predictions(tar_tr,YSO_PREDS_tr,cmap='copper',normalize='true',display_labels=YSO_labels)
plt.savefig("../Results/Figures/CM_c2d.png")

if ~Predict:
    plt.subplot()
    ConfusionMatrixDisplay.from_predictions(tar_te,YSO_PREDS_te,cmap='copper',normalize='true',display_labels=YSO_labels)
    plt.savefig(f"../Results/Figures/CM_{testset}.png")

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