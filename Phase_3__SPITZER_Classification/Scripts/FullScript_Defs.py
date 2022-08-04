import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def flag_YSE(pred1,pred2,predM,mips_ind):
    flag = []
    j = 0
    for i, p1 in enumerate(pred1):
        if j<len(mips_ind):
            if i == mips_ind[j]: # If this object is a MIPS object
                if p1==pred2[i]==predM[mips_ind[j]]:
                    flag.append(0) # All three agree
                elif p1==pred2[i] or p1==predM[mips_ind[j]] or pred2[i]==predM[mips_ind[j]]:
                    flag.append(1) # 2/3 agree 
                else:
                    flag.append(2)
            else: # If object does not have MIPS data
                if p1==pred2[i]:
                    flag.append(1) # 2/3 agree
                else:
                    flag.append(2)
            j += 1
        else: # No more possible MIPS data, continue with only three
            if p1==pred2[i]:
                flag.append(1) # 2/3 agree
            else:
                flag.append(2)
    return flag

def flag_YSO(pred1,pred2,predM,mips_ind):
    flag = []
    j = 0
    for i, p1 in enumerate(pred1):
        if j<len(mips_ind):
            if i == mips_ind[j]: # If this object is a MIPS object
                if p1==pred2[i]==predM[mips_ind[j]]:
                    flag.append(0) # All three agree
                elif p1==pred2[i] or p1==predM[mips_ind[j]] or pred2[i]==predM[mips_ind[j]]:
                    flag.append(1) # 2/3 agree 
                else:
                    flag.append(2)
            else: # If object does not have MIPS data
                if p1==pred2[i]:
                    flag.append(1) # 2/3 agree
                else:
                    flag.append(2)
            j += 1
        else: # No more possible MIPS data, continue with only three
            if p1==pred2[i]:
                flag.append(1) # 2/3 agree
            else:
                flag.append(2)
    return flag

def flag_ALL(pred_IR,pred_Y,pred_IR_2,predM,mips_ind):
    flag = []
    j = 0
    for i, p1 in enumerate(pred_IR):
        if j<len(mips_ind):
            if i == mips_ind[j]: # If this object is a MIPS object
                if p1==pred_Y[i]==pred_IR_2[i]==predM[mips_ind[j]]:
                    flag.append(0) # All four agree
                elif p1==pred_Y[i]==pred_IR_2[i] or pred_Y[i]==pred_IR_2[i]==predM[mips_ind[j]] or p1==pred_IR_2[i]==predM[mips_ind[j]]\
                    or pred_Y[i]==1==predM[mips_ind[j]]:
                    flag.append(1) # 3/4 agree
                elif p1==pred_IR_2[i] or p1==pred_Y[i] or p1==predM[mips_ind[j]] or pred_IR_2[i]==predM[mips_ind[j]] or \
                    pred_Y[i]==pred_IR_2[i] or pred_Y[i]==predM[mips_ind[j]]:
                    flag.append(2) # 3/4 agree 
                else:
                    flag.append(3)
            else: # If object does not have MIPS data
                if p1==pred_Y[i]==pred_IR_2[i]:
                        flag.append(1) # All three agree      
                elif p1==pred_IR_2[i] or p1==pred_Y[i] or pred_Y[i]==pred_IR_2[i]:
                    flag.append(2) # 2/3 agree
                else:
                    flag.append(3)
            j += 1
        else: # No more possible MIPS data, continue with only three
            if p1==pred_Y[i]==pred_IR_2[i]:
                    flag.append(1) # All three agree      
            elif p1==pred_IR_2[i] or p1==pred_Y[i] or pred_Y[i]==pred_IR_2[i]:
                flag.append(2) # 2/3 agree
            else:
                flag.append(3)
    return flag

def predbyflag(flags, alpha, MLP_pred, MLP_pred_2, YSO_pred, MLP_pred_M,mips_ind,ClassIII):
    pred = []
    for i, flag in enumerate(flags):
        if flag == 0:
            pred.append(int(MLP_pred[i]))
        elif flag == 1:
            if int(MLP_pred[i])==int(MLP_pred_2[i]):
                pred.append(int(MLP_pred[i]))
            else:
                pred.append(MLP_pred_M[np.where(mips_ind==i)])
        elif flag == 2:
            if int(MLP_pred[i])==int(MLP_pred_2[i]) or MLP_pred[i]==YSO_pred[i]:
                pred.append(int(MLP_pred[i]))
            elif MLP_pred_2[i]==YSO_pred[i]:
                pred.append(MLP_pred_2[i])
            else:
                pred.append(MLP_pred_M[np.where(mips_ind==i)])
        elif flag == 3:
            pred.append(int(MLP_pred[i]))
    if ClassIII:
        ciii = np.where(np.array(pred)==3)[0]
        for i in ciii:
            if alpha[i] < -2:
                pred[i] = 5
    return pred
    
def predbyflag_YSE(flags, alpha, MLP_pred, MLP_pred_2, MLP_pred_M,mips_ind,ClassIII):
    pred = []
    for i, flag in enumerate(flags):
        if flag == 0:
            pred.append(int(MLP_pred[i]))
        elif flag == 1:
            if int(MLP_pred[i])==int(MLP_pred_2[i]):
                pred.append(int(MLP_pred[i]))
            else:
                pred.append(MLP_pred_M[np.where(mips_ind==i)])
        elif flag == 2:
            pred.append(int(MLP_pred[i]))
    if ClassIII:
        ciii = np.where(np.array(pred)==3)[0]
        for i in ciii:
            if alpha[i] < -2:
                pred[i] = 5
    return pred

def predbyflag_YSO(flags, alpha, MLP_pred, YSO_pred, YSO_pred_M, mips_ind, ClassIII):
    pred = []
    for i, flag in enumerate(flags):
        if flag == 0:
            pred.append(int(YSO_pred[i]))
        elif flag == 1:
            if int(MLP_pred[i])==int(YSO_pred[i]):
                pred.append(int(YSO_pred[i]))
            else:
                pred.append(int(YSO_pred_M[np.where(mips_ind==i)]))
        elif flag == 2:
            pred.append(int(YSO_pred[i]))
    if ClassIII:
        ciii = np.where(np.array(pred)==3)[0]
        for i in ciii:
            if alpha[i] < -2:
                pred[i] = 5
    return pred

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
    plt.scatter(Y[np.where(flag==3)[0], 0], Y[np.where(flag==3)[0], 1], c="red",marker='x',label='Insecure')
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
