import numpy as np
import pandas as pd
from custom_dataloader import replicate_data_single
import matplotlib.pyplot as plt
from PRF import prf
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,classification_report,f1_score,recall_score,precision_score

import warnings
warnings.filterwarnings('ignore')

date = 'June192023'
dao = pd.read_csv(f'DAOPHOT_Catalog_{date}.csv')
dao_IR = pd.read_csv(f'DAOPHOT_Catalog_{date}_IR.csv')
dao_aug = pd.read_csv(f"Augmented_data_prob_{date}.csv")

date = 'DAOPHOT_'+ date
cont = True
amounts_te = []

fcd_columns = [c for c in dao_aug.columns if c[0] == "f" or c[0]=='δ'or c[0]=='S']
errs = ["e_"+f for f in fcd_columns]
bands = fcd_columns+errs
tars = ['Prob', 'Class']


def get_best_prf(tr, va, te):
    
    seed = int(np.random.default_rng().random()*np.random.default_rng().random()*10000)
    inp_tr, tar_tr = replicate_data_single(tr[:,:-1],tr[:,-1],amounts=[len(tr[:,-1][tr[:,-1]==0]),len(tr[:,-1][tr[:,-1]==1])],seed=seed)
    inp_va, tar_va = replicate_data_single(va[:,:-1],va[:,-1],amounts=[len(va[:,-1][va[:,-1]==0]),len(va[:,-1][va[:,-1]==1])],seed=seed)
    # inp_tr, tar_tr = tr[:,:-1],tr[:,-1]
    # inp_va, tar_va = va[:,:-1],va[:,-1]
    inp_tr, prob_tr = inp_tr[:,:-1], inp_tr[:,-1]
    inp_va, prob_va = inp_va[:,:-1], inp_va[:,-1]
    dy_tr = np.array([np.array([x,1-x]) for x in prob_tr])
    # dy_va = np.array([np.array([x,1-x]) for x in prob_va])


    inds = range(0,int(np.shape(inp_tr)[1]/2))
    e_inds = range(int(np.shape(inp_tr)[1]/2),np.shape(inp_tr)[1])
    prf_cls = prf(n_estimators=100, bootstrap=False, keep_proba=0.5)
    prf_cls.fit(X=inp_tr[:,inds], dX=inp_tr[:,e_inds], py=dy_tr)
    pred_va = prf_cls.predict(X=inp_va[:,inds],dX=inp_va[:,e_inds])
    pred_tr = prf_cls.predict(inp_tr[:,inds],inp_tr[:,e_inds])
    
    max_f1 = f1_score(tar_va,pred_va,average=None)[0]

    inp_te = te
    pred_te = prf_cls.predict_proba(X=inp_te[:,inds], dX=inp_te[:,e_inds]) 
    num_yso = len(np.array(pred_te)[np.transpose(pred_te)[0]>0.5])

    # print("F1-Scores:\n Training is ",f1_score(tar_tr,pred_tr,average=None)[0],"\n Validation (SPICY) is ",f1_score(tar_va,pred_va,average=None)[0])

    return np.transpose(pred_te)[0], num_yso, max_f1



def get_best_rf(tr, va, te):
    
    seed = int(np.random.default_rng().random()*np.random.default_rng().random()*10000)
    inp_tr, tar_tr = replicate_data_single(tr[:,:-1],tr[:,-1],amounts=[len(tr[:,-1][tr[:,-1]==0]),len(tr[:,-1][tr[:,-1]==1])],seed=seed)
    inp_va, tar_va = replicate_data_single(va[:,:-1],va[:,-1],amounts=[len(va[:,-1][va[:,-1]==0]),len(va[:,-1][va[:,-1]==1])],seed=seed)
    # inp_tr, tar_tr = tr[:,:-1],tr[:,-1]
    # inp_va, tar_va = va[:,:-1],va[:,-1]
    inp_tr, prob_tr = inp_tr[:,:-1], inp_tr[:,-1]
    inp_va, prob_va = inp_va[:,:-1], inp_va[:,-1]


    inds = range(0,int(np.shape(inp_tr)[1]/2))
    e_inds = range(int(np.shape(inp_tr)[1]/2),np.shape(inp_tr)[1])

    rf_cls = RandomForestClassifier(n_estimators=100)
    rf_cls.fit(inp_tr, tar_tr)
    pred_va = rf_cls.predict(inp_va)
    pred_tr = rf_cls.predict(inp_tr)
    
    max_f1 = f1_score(tar_va,pred_va,average=None)[0]

    inp_te = te
    pred_te = rf_cls.predict_proba(inp_te) 
    num_yso = len(pred_te[pred_te[:,0]>0.5])

    # print("F1-Scores:\n Training is ",f1_score(tar_tr,pred_tr,average=None)[0],"\n Validation (SPICY) is ",f1_score(tar_va,pred_va,average=None)[0])

    return pred_te[:,0], num_yso, max_f1




# Test once
# _, n, f1 = get_best_prf(tr=dao_aug[fcd_columns+errs+tars].copy().values,va=dao_IR[fcd_columns+errs+tars].copy().values,te=dao[fcd_columns+errs].copy().values)
# print(n, f1)



# -------------------------------------------
# Parallelize
prf_inds = np.array([dao_aug[fcd_columns+errs+tars].copy().values,dao_IR[fcd_columns+errs+tars].copy().values,dao[fcd_columns+errs].copy().values])
rf_inds = np.array([dao_aug[fcd_columns+errs+tars].copy().dropna().values,dao_IR[fcd_columns+errs+tars].copy().dropna().values,dao[fcd_columns+errs].copy().dropna().values])
print("Starting bootstrapping!")
import multiprocess as mp
import time
tic = time.perf_counter()
n = 100

with mp.Pool(6) as pool:
    ans_prf = pool.starmap(get_best_prf,[prf_inds] * n)
    ans_rf = pool.starmap(get_best_rf,[rf_inds] * n)

toc = time.perf_counter()
print(f"Completed the bootstrapping in {(toc - tic)/60:0.2f} minutes!\n\n")

pred_tes_prf = list(map(list, zip(*ans_prf)))[0]
num_yso_prf = list(map(list, zip(*ans_prf)))[1]
max_f1_prf = list(map(list, zip(*ans_prf)))[2]

pred_tes_rf = list(map(list, zip(*ans_rf)))[0]
num_yso_rf = list(map(list, zip(*ans_rf)))[1]
max_f1_rf = list(map(list, zip(*ans_rf)))[2]

np.savetxt("Data/Num_YSOs_RF"+date, num_yso_rf)
np.savetxt("Data/Max_f1s_RF"+date, max_f1_rf)
np.savetxt("Data/Num_YSOs_PRF"+date, num_yso_prf)
np.savetxt("Data/Max_f1s_PRF"+date, max_f1_prf)











#-----------------------------------------

# To determine classification, use mean of each row to determine probability of that object being a star. 
# If the probability to be a star is less than 50%, then the object is a YSO with probability 1-mean
# Make two columns: final class, and probability of being that class
p_yso_rf = np.nanmean(pred_tes_rf,axis=0)
preds_rf = np.zeros(len(p_yso_rf))
preds_rf[p_yso_rf<0.5] = 1 
print("Number of YSOs with prob>50\% (rf):",len(preds_rf[p_yso_rf>0.5]))
p_yso_prf = np.nanmean(pred_tes_prf,axis=0)
preds_prf = np.zeros(len(p_yso_prf))
preds_prf[p_yso_prf<0.5] = 1 
print("Number of YSOs with prob>50\% (prf):",len(preds_prf[p_yso_prf>0.5]))


# Make and save predictions/probabilities in csv
CC_Webb_Classified = pd.DataFrame()

CC_Webb_Classified['RA'] = dao['RA']
CC_Webb_Classified['DEC'] = dao[['DEC']].values
# CC_Webb_Classified['size'] = webb_inp[['size']].values
CC_Webb_Classified[np.array(bands)] = dao[bands].values
CC_Webb_Classified['Class_PRF'] = preds_prf
CC_Webb_Classified['Prob_PRF'] = p_yso_prf

CC_Webb_Classified['Class_RF'] = [-99.999]*len(dao)
CC_Webb_Classified['Prob_RF'] = [-99.999]*len(dao)
CC_Webb_Classified.loc[dao.dropna().index,'Class_RF'] = preds_rf
CC_Webb_Classified.loc[dao.dropna().index,'Prob_RF'] = p_yso_rf

# Add IR classifications for ease of comparison



CC_Webb_Classified.to_csv(f"CC_Classified_{date}.csv",index=False)
