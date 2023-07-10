import numpy as np
import pandas as pd
from custom_dataloader import replicate_data_single
import matplotlib.pyplot as plt
from PRF import prf
from sklearn.ensemble import RandomForestClassifier
from astropy.coordinates import match_coordinates_sky,SkyCoord
import astropy.units as u
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

date = 'June192023'
dao = pd.read_csv(f'DAOPHOT_Catalog_{date}.csv')
dao_IR = pd.read_csv(f'DAOPHOT_Catalog_{date}_IR.csv')
dao_aug = pd.read_csv(f"Augmented_data_prob_{date}.csv")
# dao_aug = pd.read_csv("Test_Delta_fitted_class.csv")

date = 'DAOPHOT_'+ date
cont = True
amounts_te = []

filters = [c for c in dao_aug.columns if ((c[0] == "f") and ('-' not in c))]
fcd_columns = [c for c in dao_aug.columns if ((c[0] == "f") and ('-' in c)) or c[0]=='δ' or c[0]=='(' or c=='Sum1' or c[0]=='s']# and ('f470n' not in c) and ('f187n' not in c)]#or c=='Sum1'
print(fcd_columns)
errs = ["e_"+f for f in fcd_columns]
bands = fcd_columns+errs
tars = ['Prob', 'Class']

# Normalize data
max_fcde = np.array([np.nanmax(dao[c].values) for c in fcd_columns])
dao_aug_norm = dao_aug.copy()
dao_aug_norm[fcd_columns] = dao_aug_norm[fcd_columns].values/max_fcde
dao_aug_norm[errs] = dao_aug_norm[errs].values/max_fcde

dao_norm = dao.copy()
dao_norm[fcd_columns] = dao_norm[fcd_columns].values/max_fcde
dao_norm[errs] = dao_norm[errs].values/max_fcde

dao_IR_norm = dao_IR.copy()
dao_IR_norm[fcd_columns] = dao_IR_norm[fcd_columns].values/max_fcde
dao_IR_norm[errs] = dao_IR_norm[errs].values/max_fcde


def get_best_prf(tr, va, te):
    
    seed = int(np.random.default_rng().random()*np.random.default_rng().random()*10000)
    # tr, _ = train_test_split(tr,train_size=.9,random_state=seed)
    # va, _ = train_test_split(va,train_size=1.)
    inp_tr, tar_tr = replicate_data_single(tr[:,:-1],tr[:,-1],amounts=[len(tr[:,-1][tr[:,-1]==0]),len(tr[:,-1][tr[:,-1]==1])],seed=seed)
    inp_va, tar_va = replicate_data_single(va[:,:-1],va[:,-1],amounts=[len(va[:,-1][va[:,-1]==0]),len(va[:,-1][va[:,-1]==1])],seed=seed)
    inp_tr, tar_tr = tr[:,:-1],tr[:,-1]
    # inp_va, tar_va = va[:,:-1],va[:,-1]
    inp_tr, prob_tr = inp_tr[:,:-1], inp_tr[:,-1]
    inp_va, _ = inp_va[:,:-1], inp_va[:,-1]
    dy_tr = np.array([np.array([x,1-x]) for x in prob_tr])
    # dy_va = np.array([np.array([x,1-x]) for x in prob_va])


    inds = range(0,int(np.shape(inp_tr)[1]/2))
    e_inds = range(int(np.shape(inp_tr)[1]/2),np.shape(inp_tr)[1])
    prf_cls = prf(n_estimators=100, bootstrap=False, keep_proba=0.5)
    prf_cls.fit(X=inp_tr[:,inds], dX=inp_tr[:,e_inds], y=tar_tr)#py=dy_tr)#
    # prf_cls.fit(X=inp_tr[:,inds], dX=inp_tr[:,e_inds], py=dy_tr)
    pred_va = prf_cls.predict(X=inp_va[:,inds],dX=inp_va[:,e_inds])
    # pred_tr = prf_cls.predict(inp_tr[:,inds],inp_tr[:,e_inds])
    
    max_f1 = f1_score(tar_va,pred_va,average=None)[0]

    pred_te = prf_cls.predict_proba(X=te[:,inds], dX=te[:,e_inds]) 
    num_yso = len(np.array(pred_te)[np.transpose(pred_te)[0]>0.5])

    # print("F1-Scores:\n Training is ",f1_score(tar_tr,pred_tr,average=None)[0],"\n Validation (SPICY) is ",f1_score(tar_va,pred_va,average=None)[0])

    return np.transpose(pred_te)[0], num_yso, max_f1



def get_best_rf(tr, va, te):
    
    seed = int(np.random.default_rng().random()*np.random.default_rng().random()*10000)
    # tr, _ = train_test_split(tr,train_size=.9,random_state=seed)
    # va, _ = train_test_split(va,train_size=1.)
    inp_tr, tar_tr = replicate_data_single(tr[:,:-1],tr[:,-1],amounts=[len(tr[:,-1][tr[:,-1]==0]),len(tr[:,-1][tr[:,-1]==1])],seed=seed)
    inp_va, tar_va = replicate_data_single(va[:,:-1],va[:,-1],amounts=[len(va[:,-1][va[:,-1]==0]),len(va[:,-1][va[:,-1]==1])],seed=seed)
    inp_tr, tar_tr = tr[:,:-1],tr[:,-1]
    # inp_va, tar_va = va[:,:-1],va[:,-1]
    inp_tr = inp_tr[:,:-1] # Remove probability
    inp_va = inp_va[:,:-1] # Remove probability

    rf_cls = RandomForestClassifier(n_estimators=100)
    rf_cls.fit(inp_tr, tar_tr)
    pred_va = rf_cls.predict(inp_va)
    # pred_tr = rf_cls.predict(inp_tr)
    
    max_f1 = f1_score(tar_va,pred_va,average=None)[0]

    pred_te = rf_cls.predict_proba(te) 
    num_yso = len(pred_te[pred_te[:,0]>0.5])

    # print("F1-Scores:\n Training is ",f1_score(tar_tr,pred_tr,average=None)[0],"\n Validation (SPICY) is ",f1_score(tar_va,pred_va,average=None)[0])

    return pred_te[:,0], num_yso, max_f1




# Test once
# _, n, f1 = get_best_rf(dao_aug[fcd_columns+errs+tars].copy().dropna().values,dao_IR[fcd_columns+errs+tars].copy().dropna().values,dao[fcd_columns+errs].copy().dropna().values)
# print(n, f1)



# -------------------------------------------
# Parallelize
prf_inds = np.array([dao_aug_norm[fcd_columns+errs+tars].copy().values,dao_IR_norm[fcd_columns+errs+tars].copy().values,dao_norm[fcd_columns+errs].copy().values])
rf_inds = np.array([dao_aug_norm[fcd_columns+errs+tars].copy().dropna().values,dao_IR_norm[fcd_columns+errs+tars].copy().dropna().values,dao_norm[fcd_columns+errs].copy().dropna().values])
print("Starting bootstrapping!")
import multiprocess as mp
import time
tic = time.perf_counter()
n = 50

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

# np.savetxt("Data/Num_YSOs_RF"+date, num_yso_rf)
# np.savetxt("Data/Max_f1s_RF"+date, max_f1_rf)
# np.savetxt("Data/Num_YSOs_PRF"+date, num_yso_prf)
# np.savetxt("Data/Max_f1s_PRF"+date, max_f1_prf)





#-----------------------------------------

# To determine classification, use mean of each row to determine probability of that object being a star. 
# If the probability to be a star is less than 50%, then the object is a YSO with probability 1-mean
# Make two columns: final class, and probability of being that class
p_yso_rf = np.nanmean(pred_tes_rf,axis=0)
preds_rf = np.zeros(len(p_yso_rf))
preds_rf[p_yso_rf<=0.5] = 1 
print("Number of YSOs with prob>50\% (rf):",len(preds_rf[p_yso_rf>0.5]))
p_yso_prf = np.nanmean(pred_tes_prf,axis=0)
preds_prf = np.zeros(len(p_yso_prf))
preds_prf[p_yso_prf<=0.5] = 1 
print("Number of YSOs with prob>50\% (prf):",len(preds_prf[p_yso_prf>0.5]))


# Make and save predictions/probabilities in csv
CC_Webb_Classified = dao.copy()[['Index','RA','DEC']+[c for c in dao_aug.columns if (c[0] == "f" or c[0]=='δ'or c=='Sum1' or c[0]=='e'or c[0]=='(')]]


# CC_Webb_Classified['Index'] = dao['Index']
# CC_Webb_Classified['RA'] = dao['RA']
# CC_Webb_Classified['DEC'] = dao[['DEC']].values
# CC_Webb_Classified[np.array(bands)] = dao[bands].values
CC_Webb_Classified['Class_PRF'] = preds_prf
CC_Webb_Classified['Prob_PRF'] = p_yso_prf

CC_Webb_Classified['Class_RF'] = [-99.999]*len(dao)
CC_Webb_Classified['Prob_RF'] = [-99.999]*len(dao)
CC_Webb_Classified.loc[dao[fcd_columns+errs].dropna().index,'Class_RF'] = preds_rf
CC_Webb_Classified.loc[dao[fcd_columns+errs].dropna().index,'Prob_RF'] = p_yso_rf

# CC_Webb_Classified.to_csv(f"CC_Classified_{date}.csv",index=False)
# print("Saved CC_Webb_Classified in case of failure at next step.")

# Add IR classifications for ease of comparison
ind, sep, _ = match_coordinates_sky(SkyCoord(dao_IR.RA,dao_IR.DEC,unit=u.deg),SkyCoord(CC_Webb_Classified.RA,CC_Webb_Classified.DEC,unit=u.deg))

CC_Webb_Classified['Init_Class'] = [np.nan]*len(CC_Webb_Classified)
CC_Webb_Classified.loc[ind,'Init_Class'] = dao_IR.Class.values
CC_Webb_Classified['Survey'] = [np.nan]*len(CC_Webb_Classified)
CC_Webb_Classified.loc[ind,'Survey'] = dao_IR.Survey.values
CC_Webb_Classified['SPICY_ID'] = [np.nan]*len(CC_Webb_Classified)
CC_Webb_Classified.loc[ind,'SPICY_ID'] = dao_IR.SPICY_ID.values

CC_Webb_Classified.to_csv(f"CC_Classified_{date}.csv",index=False)
print("Classification finished and comparison to previous work added!")


print("RF Classification Report")
print(classification_report(CC_Webb_Classified.dropna(subset=['Init_Class']+filters).Init_Class,CC_Webb_Classified.dropna(subset=['Init_Class']+filters).Class_RF))

print("PRF Classification Report")
print(classification_report(CC_Webb_Classified.dropna(subset='Init_Class').Init_Class,CC_Webb_Classified.dropna(subset='Init_Class').Class_PRF))