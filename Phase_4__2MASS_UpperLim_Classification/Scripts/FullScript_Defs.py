import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from NN_Defs import TwoLayerMLP, test, MLP_data_setup
from scipy import stats


def predict_yse(inp_tr,tar_tr,inp_te,tar_te,bands,device):
    if bands[0]=='mag_J':
        bands[0] = 'mag_2M1'
    b = [idx[slice(-3,-1)] for idx in bands if (idx[-1] == '1' and idx[0] != 'e')]
    # Add in only networks for which you have data:
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
        band_ind = np.where(np.isin(bands,band))[0]
        I2M_train, I2M_valid, I2M_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
        NN_I2M = TwoLayerMLP(len(band_ind), 20, 3)
        NN_I2M.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_17_LR_Reduce_YSE/IRAC_MIPS_2MASS_alphaLR_0.001_MO__NEUR_20_Settings", map_location=device))
        # Test MLP
        I2M_preds_tr = test(NN_I2M, I2M_train, device)
        I2M_preds_te = test(NN_I2M, I2M_test, device)
    else:
        I2M_preds_tr = [np.nan]*len(IR_preds_tr)
        I2M_preds_te = [np.nan]*len(IR_preds_te)
    if "MP" in b[-1]:
        # Test IRAC and MIPS
        band = [idx for idx in bands if (idx[-2].lower() == 'R'.lower() or idx[-2].lower() == 'P'.lower() or idx == 'alpha')]
        band_ind = np.where(np.isin(bands,band))[0]
        IM_train, IM_valid, IM_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
        NN_IM = TwoLayerMLP(len(band_ind), 10, 3)
        NN_IM.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_17_LR_Reduce_YSE/IRAC_MIPS_alphaLR_0.001_MO__NEUR_10_Settings", map_location=device))
        # Test MLP
        IM_preds_tr = test(NN_IM, IM_train, device)
        IM_preds_te = test(NN_IM, IM_test, device)
    else:
        IM_preds_tr = [np.nan]*len(IR_preds_tr)
        IM_preds_te = [np.nan]*len(IR_preds_te)
    if "2M" in b[0]:
        # Test IRAC and 2MASS
        band = [idx for idx in bands if (idx[-2].lower() != 'P'.lower())]
        band_ind = np.where(np.isin(bands,band))[0]
        I2_train, I2_valid, I2_test = MLP_data_setup(inp_tr[:,band_ind], tar_tr, inp_te[:,band_ind], tar_te, inp_te[:,band_ind], tar_te)
        NN_I2 = TwoLayerMLP(len(band_ind), 20, 3)
        NN_I2.load_state_dict(torch.load("../Results/Best_Results/c2d_quality_17_LR_Reduce_YSE/IRAC_2MASS_alphaLR_0.001_MO__NEUR_20_Settings", map_location=device))
        # Test MLP
        I2_preds_tr = test(NN_I2, I2_train, device)
        I2_preds_te = test(NN_I2, I2_test, device)
    else:
        I2_preds_tr = [np.nan]*len(IR_preds_tr)
        I2_preds_te = [np.nan]*len(IR_preds_te)



    # Determine matching predictions by mode
    # Combine predictions into nxm grid. n - number of objects, m - number of classifications used\
    Preds_tr = np.c_[IR_preds_tr,IM_preds_tr,I2_preds_tr,I2M_preds_tr]
    Preds_te = np.c_[IR_preds_te,IM_preds_te,I2_preds_te,I2M_preds_te]


    flags_tr = np.array(flag(Preds_tr)).ravel()
    flags_te = np.array(flag(Preds_te)).ravel()
    return Preds_tr, Preds_te, flags_tr, flags_te

def flag(arr):
    # cols = np.shape(arr[0][~np.isnan(arr[0])])[0]
    arr_f = []
    for n in arr:
        if len(np.unique(n[~np.isnan(n)]))<=2: # There is 2 or less unique values. If only have 2 methods this will always return Secure
            arr_f.append(0) # Secure
        else:
            arr_f.append(1) # Insecure
    return arr_f

def tsne_plot(inp,pred,flag,type,three=False):
    n_components = 2
    tsne = TSNE(
        n_components=n_components,
        init="random",
        random_state=11,
        perplexity=30,
        learning_rate="auto",
        n_iter=300)
    Y = tsne.fit_transform(inp)

    lw = 0.3
    plt.rcParams["scatter.edgecolors"]  = 'k'
    plt.rcParams["font.family"] = "times"
    plt.title("t-SNE of predictions with insecure objects marked")
    if three:
        plt.scatter(Y[np.where(pred==5)[0], 0], Y[np.where(pred==5)[0], 1], c="green",marker='*',linewidths=lw,label='Stars')
        plt.scatter(Y[np.where(pred==4)[0], 0], Y[np.where(pred==4)[0], 1], c="gold",marker='s',linewidths=lw,label='EG')
        plt.scatter(Y[np.where(pred==3)[0], 0], Y[np.where(pred==3)[0], 1], c="mediumseagreen",linewidths=lw,label='YSO - Class III')
    else:
        plt.scatter(Y[np.where(pred==4)[0], 0], Y[np.where(pred==4)[0], 1], c="green",marker='*',linewidths=lw,label='Stars')
        plt.scatter(Y[np.where(pred==3)[0], 0], Y[np.where(pred==3)[0], 1], c="gold",marker='s',linewidths=lw,label='EG')
    plt.scatter(Y[np.where(pred==2)[0], 0], Y[np.where(pred==2)[0], 1], c="mediumspringgreen",linewidths=lw,label='YSO - Class II')
    plt.scatter(Y[np.where(pred==1)[0], 0], Y[np.where(pred==1)[0], 1], c="mediumaquamarine",linewidths=lw,label='YSO - Class FS')
    plt.scatter(Y[np.where(pred==0)[0], 0], Y[np.where(pred==0)[0], 1], c="aquamarine",linewidths=lw,label='YSO - Class I')
    plt.legend()
    plt.savefig(f"../Results/Figures/t-SNE_final_{type}_WO_InSecure.png",dpi=300)
    plt.scatter(Y[np.where(flag==1)[0], 0], Y[np.where(flag==1)[0], 1], c="red",marker='x',label='Insecure')
    plt.legend()
    plt.savefig(f"../Results/Figures/t-SNE_final_{type}_W_InSecure.png",dpi=300)
    plt.close()

def plot_hist(inp,pred,type,ClassIII):
    alpha = inp[:,-1]
    plt.rcParams["font.family"] = "times"
    plt.title("Spectral index of predicted YSO stages")
    color = ["aquamarine","mediumaquamarine","mediumspringgreen","mediumseagreen"]
    stage = ["YSO - Class I","YSO - Class FS","YSO - Class II","YSO - Class III"]
    bins=np.linspace(-6,6,60)
    if ClassIII:
        n = 3
    else:
        n=2
    for i,p in enumerate(list(range(0, n+1))):
        plt.hist(alpha[np.where(pred==p)[0]],bins,color=color[i],label=stage[i])
    mu = np.mean(alpha[np.where(pred<=n)[0]])
    sig = np.std(alpha[np.where(pred<=n)[0]])
    binwidth=0.2
    yM = binwidth*len(pred[np.where(pred<=n)])
    bins_gaus = np.linspace(-6,6,600)
    plt.plot(bins_gaus, yM*(1/(sig * np.sqrt(2 * np.pi)) * np.exp( - (bins_gaus - mu)**2 / (2 * sig**2)) ),linewidth=2, color='k')
    plt.vlines(0.3,ymax=yM*(1/(sig * np.sqrt(2 * np.pi)) * np.exp( - (0.3 - mu)**2 / (2 * sig**2))),ymin=0,color='k')
    plt.vlines(-0.3,ymax=yM*1/(sig * np.sqrt(2 * np.pi)) * np.exp( - (-0.3 - mu)**2 / (2 * sig**2)),ymin=0,color='k')
    plt.vlines(-1.6,ymax=yM*1/(sig * np.sqrt(2 * np.pi)) * np.exp( - (-1.6 - mu)**2 / (2 * sig**2)),ymin=0,color='k')
    plt.xlabel("Spectral index α")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(f"../Results/Figures/alph_hist_{type}.png",dpi=300)
    plt.close()
