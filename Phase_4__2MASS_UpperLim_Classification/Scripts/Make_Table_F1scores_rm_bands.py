import numpy as np
import pandas as pd
from custom_dataloader import replicate_data_single
import matplotlib.pyplot as plt
from PRF import prf

from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,classification_report,f1_score,recall_score,precision_score

import warnings
warnings.filterwarnings('ignore')

date = 'Dec192022'

webb_inp = pd.read_csv('../../NGC_3324/CC_JWST_NIRCAM_MIRI_Full_'+date+'_2pt_vegamag_flux.csv')
all_inp = pd.read_csv('CC_Webb_NIRCam_MIRI_Spitz_2m_w_SPICY_Preds_'+date+'.csv')

cont = True
amounts_te = []
# 090, 187, 200, 335, 444, 470, 770, 1130, 1280, 1800
# inds = (0,2,4,6,8,10,12,14,16,18)

bands = [idx for idx in webb_inp.columns.values if (idx[:3].lower() == 'iso'.lower() and idx[10]=='v')]
# print(np.array(bands)[np.array(inds)])

input_webb = all_inp[bands].to_numpy()
tar_webb = all_inp[['SPICY_Class_0/1']].to_numpy()


def get_best_prf(band,i1=np.nan,i2=np.nan,i3=np.nan,i4=np.nan,i5=np.nan,i6=np.nan,i7=np.nan,i8=np.nan,i9=np.nan,i10=np.nan):
    inds = np.array([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10])
    inds = inds[~np.isnan(inds)]
    inds = np.array([int(i) for i in inds])
    err_inds = np.array([int(i)+1 for i in inds])
    max_f1_va = 0.5
    for i in np.arange(0,1000,2):
        seed = int(np.random.default_rng().random()*np.random.default_rng().random()*10000)
        inp_tr, tar_tr = replicate_data_single(input_webb,tar_webb,amounts=[len(tar_webb[tar_webb==0]),len(tar_webb[tar_webb==0])*3],seed=seed)
        inp_va, tar_va = replicate_data_single(input_webb,tar_webb,amounts=[len(tar_webb[tar_webb==0]),len(tar_webb[tar_webb==1])],seed=seed)

        prf_cls = prf(n_estimators=100, bootstrap=False, keep_proba=0.75)
        prf_cls.fit(X=inp_tr[:,inds], dX=inp_tr[:,err_inds], y=tar_tr)

        pred_va = prf_cls.predict(X=inp_va[:,inds],dX=inp_va[:,err_inds])

        if (f1_score(tar_va,pred_va,average=None)[0] > max_f1_va):
            
            max_prf = prf_cls
            max_f1_va = f1_score(tar_va,pred_va,average=None)[0]

            pred_tr = prf_cls.predict(X=inp_tr[:,inds],dX=inp_tr[:,err_inds])
            max_f1_tr = f1_score(tar_tr,pred_tr,average=None)[0]
    
    
    inp_te = webb_inp[np.array(bands)].to_numpy()
    pred_te_max = max_prf.predict(X=inp_te[:,inds], dX=inp_te[:,err_inds]) 
    num_yso = len(pred_te_max[pred_te_max==0]) 

    return band,num_yso, max_f1_tr, max_f1_va




##-------------------------------------------
# Parallelize

all_inds = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
print("Starting bootstrapping!")
import multiprocess as mp
import time
tic = time.perf_counter()
iters = [['All bands present', 0, 2, 4, 6, 8, 10, 12, 14, 16, 18],['F470N', 0, 2, 4, 6, 8, 12, 14, 16, 18],['F444W',0, 2, 4, 6, 10, 12, 14, 16, 18],['F335M', 0, 2, 4, 8, 10, 12, 14, 16, 18],\
    ['F187N', 0, 4, 6, 8, 10, 12, 14, 16, 18],['F090W', 2, 4, 6, 8, 10, 12, 14, 16, 18],['F200W',0, 2, 6, 8, 10, 12, 14, 16, 18],['F1280W', 0, 2, 4, 6, 8, 10, 12, 14, 18],['F1130W', 0, 2, 4, 6, 8, 10, 12, 16, 18],\
        ['F770W', 0, 2, 4, 6, 8, 10, 14, 16, 18],['F1800W', 0, 2, 4, 6, 8, 10, 12, 14, 16],['N bands removed', 0, 4, 6, 8, 12, 14, 16, 18],['MIRI bands removed',0, 2, 4, 6, 8, 10]]
with mp.Pool(6) as pool:
    ans = pool.starmap(get_best_prf,iters)
# num_yso = np.zeros(3)
# max_f1_tr = np.zeros(3)
# max_f1_va = np.zeros(3)
# i=0
# for iter in iters[:3]:
#     num_yso[i], max_f1_tr[i], max_f1_va[i] = get_best_prf(iter)
#     i = i+1


toc = time.perf_counter()
print(f"Completed the bootstrapping in {(toc - tic)/60:0.2f} minutes!\n\n")

rownames = list(map(list, zip(*ans)))[0]
num_yso = list(map(list, zip(*ans)))[1]
max_f1_tr = list(map(list, zip(*ans)))[2]
max_f1_va = list(map(list, zip(*ans)))[3]

print("Shape of num_ysos:",np.shape(num_yso))
print("Shape of max_f1:",np.shape(max_f1_tr))

f = open("F1Scores_Num_YSOs.txt",'w')

f.write("Band &  F1 \% (tr) & F1 \% (va) & \# YSOs\\\ \n")
for i in np.arange(0,len(num_yso)):
    f.write(f"{rownames[i]} & {round(max_f1_tr[i]*100)} & {round(max_f1_va[i]*100)}  & {num_yso[i]} \\\ \n")