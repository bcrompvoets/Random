import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PRF import prf
from sklearn.ensemble import RandomForestClassifier
from astropy.coordinates import match_coordinates_sky,SkyCoord
import astropy.units as u
from sklearn.metrics import f1_score,classification_report
from sklearn.utils import shuffle
from Augment_Data import EVT
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, KMeansSMOTE, SVMSMOTE
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

date = 'CC_Mar42024'
path_sv = "./"+date+'/'

# Make a new directory to save files in if it doesn't already exist.
Path(path_sv).mkdir(parents=True, exist_ok=True)

# Open the photometry catalogs (dao = full set, dao_IR = objects with classifications)
dao = pd.read_csv(f'DAOPHOT_Catalog_{date}.csv')
dao_IR = pd.read_csv(f'DAOPHOT_Catalog_{date}_IR.csv')

filt='f470n'
date = date+'_no_'+filt
date = date+'_with_10_percent_saved'

filters = [c for c in dao.columns if ((c[0] == "f") and ('-' not in c)and (filt not in c) )]#and (filt not in c) 
e_filters = ["e_"+f for f in filters]
print(filters)

# Augment data using EVT method
to_aug_x, _, to_aug_y, _ = train_test_split(dao_IR.dropna(subset=filters)[filters+e_filters].values,dao_IR.dropna(subset=filters)[['Class']].values,test_size=0.1)
to_aug = pd.DataFrame(data=to_aug_x, columns=filters+e_filters)
to_aug['Class'] = to_aug_y

print('Input dataset shape %s' % Counter(to_aug_y.ravel()))
dao_aug = EVT(inp_df=to_aug,filters=filters,num_objs=10000)
print('Resampled dataset shape %s' % Counter(dao_aug.Class.values))
# print(filters+e_filters)

# Augment data using SMOTE method
# date = date +'_ADASYN'
# X = dao_IR.dropna(subset=filters)[filters].values
# y = dao_IR.dropna(subset=filters)[['Class']].values
# print('Input dataset shape %s' % Counter(y.ravel()))
# ada = ADASYN(sampling_strategy='all',random_state=42)
# X_res, y_res = ada.fit_resample(X, y)
# print('Resampled dataset shape %s' % Counter(y_res))

# dao_aug = pd.DataFrame(data=X_res, columns=filters)
# dao_aug['Class'] = y_res
# for f in filters:
#     dao_aug['e_'+f] = [0.1]*len(dao_aug)

# Data processing:
# Add in colours and deltas 
def add_cols(dao_fcd,filters,main_trend=None):
    """Function for adding in colours for the specified filters. All filter combinations are returned."""

    for f, filt in enumerate(filters):
        for filt2 in filters[f+1:]:
            col = filt+"-"+filt2
            dao_fcd[col] = dao_fcd[filt] - dao_fcd[filt2]
            dao_fcd["e_"+col] = np.sqrt(dao_fcd["e_"+filt].values**2 + dao_fcd["e_"+filt2].values**2)
            
    return dao_fcd


dao = add_cols(dao.copy(),main_trend=dao.copy(),filters=filters)
dao_IR = add_cols(dao_IR.copy(),main_trend=dao.copy(),filters=filters)
dao_aug = add_cols(dao_aug.copy(),main_trend=dao.copy(),filters=filters)

#------------------------------------------------------
fcd_columns = [c for c in dao_aug.columns if ((c[0] != "e") and ('-' in c))]
print(fcd_columns)
errs = ["e_"+f for f in fcd_columns]
tars = ['Class']

# Normalize data
# max_fcde = np.array([np.nanmax(dao[c].values) for c in fcd_columns])
dao_aug_norm = dao_aug.copy()
# dao_aug_norm[fcd_columns] = dao_aug_norm[fcd_columns].values/max_fcde
# dao_aug_norm[errs] = dao_aug_norm[errs].values/max_fcde

dao_norm = dao.copy()
# dao_norm[fcd_columns] = dao_norm[fcd_columns].values/max_fcde
# dao_norm[errs] = dao_norm[errs].values/max_fcde

dao_IR_norm = dao_IR.copy()
# dao_IR_norm[fcd_columns] = dao_IR_norm[fcd_columns].values/max_fcde
# dao_IR_norm[errs] = dao_IR_norm[errs].values/max_fcde


def get_best_prf(tr, va, te):
    # Shuffle input data to ensure results are reproducible regardless of random seed
    inp_tr, tar_tr = shuffle(tr[:,:-1],tr[:,-1])
    inp_va, tar_va = shuffle(va[:,:-1],va[:,-1])

    inds = range(0,int(np.shape(inp_tr)[1]/2))
    e_inds = range(int(np.shape(inp_tr)[1]/2),np.shape(inp_tr)[1])

    prf_cls = prf(n_estimators=100, bootstrap=False)
    prf_cls.fit(X=inp_tr[:,inds], dX=inp_tr[:,e_inds], y=tar_tr)
    pred_va = prf_cls.predict(X=inp_va[:,inds],dX=inp_va[:,e_inds])
    
    max_f1 = f1_score(tar_va,pred_va,average=None)[0]
    pred_te = prf_cls.predict_proba(X=te[:,inds], dX=te[:,e_inds]) 
    num_yso = len(np.array(pred_te)[np.transpose(pred_te)[0]>0.5])

    return np.transpose(pred_te)[0], num_yso, max_f1, [prf_cls.feature_importances_]



def get_best_rf(tr, va, te):
    # Shuffle input data to ensure results are reproducible regardless of random seed
    inp_tr, tar_tr = shuffle(tr[:,:-1],tr[:,-1])
    inp_va, tar_va = shuffle(va[:,:-1],va[:,-1])

    rf_cls = RandomForestClassifier(n_estimators=100)
    rf_cls.fit(inp_tr, tar_tr)
    pred_va = rf_cls.predict(inp_va)
    
    max_f1 = f1_score(tar_va,pred_va,average=None)[0]
    pred_te = rf_cls.predict_proba(te) 
    num_yso = len(pred_te[pred_te[:,0]>0.5])

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
n = 100 # Number of iterations to run over

with mp.Pool(6) as pool:
    ans_prf = pool.starmap(get_best_prf,[prf_inds] * n)
    ans_rf = pool.starmap(get_best_rf,[rf_inds] * n)

toc = time.perf_counter()
print(f"Completed the bootstrapping in {(toc - tic)/60:0.2f} minutes!\n\n")

# Save the number and f1-scores for each iteration to later review to see variation
pred_tes_prf = list(map(list, zip(*ans_prf)))[0]
num_yso_prf = list(map(list, zip(*ans_prf)))[1]
max_f1_prf = list(map(list, zip(*ans_prf)))[2]
feature_importances_prf = list(map(list, zip(*ans_prf)))[3]

pred_tes_rf = list(map(list, zip(*ans_rf)))[0]
num_yso_rf = list(map(list, zip(*ans_rf)))[1]
max_f1_rf = list(map(list, zip(*ans_rf)))[2]

np.savetxt(path_sv+"Num_YSOs_RF_"+date, num_yso_rf)
np.savetxt(path_sv+"Max_f1s_RF_"+date, max_f1_rf)
np.savetxt(path_sv+"Num_YSOs_PRF_"+date, num_yso_prf)
np.savetxt(path_sv+"Max_f1s_PRF_"+date, max_f1_prf)

# Calculate the sum of feature importances across all runs
df = pd.DataFrame(feature_importances_prf[0], columns=fcd_columns)
total_importance = df.sum(axis=0)
ranked_features = pd.DataFrame(data={'Feature':fcd_columns,'Importance':total_importance})
ranked_features.sort_values('Importance',inplace=True,ascending=True)
ranked_features.to_csv(path_sv+"CC_feature_importances_"+date+".csv")

#-----------------------------------------

# To determine classification, use mean of each row to determine probability of that object being a star. 
# If the probability to be a star is less than 50%, then the object is a YSO with probability 1-mean
# Make two columns: final class, and probability of being that class
p_yso_rf = np.nanmean(pred_tes_rf,axis=0)
preds_rf = np.zeros(len(p_yso_rf))
preds_rf[p_yso_rf<0.5] = 1 
print("Number of YSOs with prob > 50\% (rf):",len(preds_rf[p_yso_rf>0.5]),len(preds_rf))
print("Standard deviation in probability of being a YSO for RF: ", np.nanmean(np.nanstd(pred_tes_rf,axis=0)))
p_yso_prf = np.nanmean(pred_tes_prf,axis=0)
preds_prf = np.zeros(len(p_yso_prf))
preds_prf[p_yso_prf<0.5] = 1 
print("Number of YSOs with prob > 50\% (prf):",len(preds_prf[p_yso_prf>0.5]))
print("Standard deviation in probability of being a YSO for PRF: ", np.nanmean(np.nanstd(pred_tes_prf,axis=0)))


# Make and save predictions/probabilities in csv
CC_Webb_Classified = dao.copy()[['Index','RA','DEC','x','y']+filters+["e_"+f for f in filters]+fcd_columns+errs]

CC_Webb_Classified['Class_PRF'] = preds_prf
CC_Webb_Classified['Prob_PRF'] = p_yso_prf

CC_Webb_Classified['Class_RF'] = [-99.999]*len(dao)
CC_Webb_Classified['Prob_RF'] = [-99.999]*len(dao)
CC_Webb_Classified.loc[dao[fcd_columns+errs].dropna().index,'Class_RF'] = preds_rf
CC_Webb_Classified.loc[dao[fcd_columns+errs].dropna().index,'Prob_RF'] = p_yso_rf

# Add Spitzer classifications for ease of comparison
ind, sep, _ = match_coordinates_sky(SkyCoord(dao_IR.RA,dao_IR.DEC,unit=u.deg),SkyCoord(CC_Webb_Classified.RA,CC_Webb_Classified.DEC,unit=u.deg))

CC_Webb_Classified['Init_Class'] = [np.nan]*len(CC_Webb_Classified)
CC_Webb_Classified.loc[ind,'Init_Class'] = dao_IR.Class.values
CC_Webb_Classified['Survey'] = [np.nan]*len(CC_Webb_Classified)
CC_Webb_Classified.loc[ind,'Survey'] = dao_IR.Survey.values
CC_Webb_Classified['SPICY_ID'] = [np.nan]*len(CC_Webb_Classified)
CC_Webb_Classified.loc[ind,'SPICY_ID'] = dao_IR.SPICY_ID.values

CC_Webb_Classified.to_csv(path_sv+f"CC_Classified_"+date+".csv",index=False)

# Un-comment to save results in CARTA-readable format
# from astropy.table import Table
# from astropy.io.votable import from_table, writeto
# print("Classification finished and comparison to previous work added!")
# tmp_votab = from_table(Table.from_pandas(CC_Webb_Classified,units={'RA':u.deg,'DEC':u.deg}))
# writeto(tmp_votab, f"Classifications_CC_{date}.xml")

# Report some basic scores for checking
print("F1-RF: ", f1_score(CC_Webb_Classified.dropna(subset=['Init_Class']+filters).Init_Class,CC_Webb_Classified.dropna(subset=['Init_Class']+filters).Class_RF), "F1-PRF: ", f1_score(CC_Webb_Classified.dropna(subset='Init_Class').Init_Class,CC_Webb_Classified.dropna(subset='Init_Class').Class_PRF))

print("RF Classification Report")
print(classification_report(CC_Webb_Classified.dropna(subset=['Init_Class']+filters).Init_Class,CC_Webb_Classified.dropna(subset=['Init_Class']+filters).Class_RF))

print("PRF Classification Report")
print(classification_report(CC_Webb_Classified.dropna(subset='Init_Class').Init_Class,CC_Webb_Classified.dropna(subset='Init_Class').Class_PRF))