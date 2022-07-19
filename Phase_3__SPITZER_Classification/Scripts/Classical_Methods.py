import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb

import numpy as np
import matplotlib.pyplot as plt
from custom_dataloader import replicate_data_single
from NN_Defs import preproc_yso
import joblib

X_tr = np.load("../Data/c2d_1k_INP.npy") # Load input data
Y_tr = np.load("../Data/c2d_1k_TAR.npy") # Load target data
X_tr = np.float32(X_tr)
Y_tr = np.float32(Y_tr)
inp_tr, tar_tr = replicate_data_single(X_tr, Y_tr, [len(np.where(Y_tr==0.)[0]),len(np.where(Y_tr==1.)[0]),len(np.where(Y_tr==2.)[0])])

X_te = np.load("../Data/NGC2264_INP.npy") # Load input data
Y_te = np.load("../Data/NGC2264_TAR.npy") # Load target data
X_te = np.float32(X_te)
Y_te = np.float32(Y_te)
inp_te, tar_te = replicate_data_single(X_te, Y_te, [len(np.where(Y_te==0.)[0]),len(np.where(Y_te==1.)[0]),len(np.where(Y_te==2.)[0])])

# Data transformation
inp_tr = np.delete(inp_tr,np.s_[8:10],axis=1)
inp_te = np.delete(inp_te,np.s_[8:10],axis=1)
# inp_tr = inp_tr[np.where(inp_tr[:,9]!=-99)[0]]
# tar_tr = tar_tr[np.where(inp_tr[:,9]!=-99)[0]]
# inp_te = inp_te[np.where(inp_te[:,9]!=-99)[0]]
# tar_te = tar_te[np.where(inp_te[:,9]!=-99)[0]]
scaler_S = StandardScaler().fit(inp_tr)
inp_tr = scaler_S.transform(inp_tr)
inp_te = scaler_S.transform(inp_te) 


tar_tr = preproc_yso(inp_tr[:,-1],tar_tr)
tar_te = preproc_yso(inp_te[:,-1],tar_te)
target_names = ["YSO - Class I","YSO - Class FS","YSO - Class II","EG", "Stars"]
# target_names = ["YSO","EG", "Stars"]


f = open("../Results/Classical_Methods_YSO_All.txt", "w")
# Initiate, fit, and predict targets with a gradient boosting classifier
gbcl = GradientBoostingClassifier(criterion='friedman_mse',max_depth=5,max_features='log2',\
     n_estimators=150,n_iter_no_change=5,subsample=1.0,warm_start=False)
gbcl.fit(inp_tr,tar_tr.ravel())
gb_preds_tr = gbcl.predict(inp_tr)
gb_preds_te = gbcl.predict(inp_te)
f.write("GB c2d Results\n")
f.write(classification_report(tar_tr,gb_preds_tr,target_names=target_names))
f.write("\nGB NGC 2264 Results\n")
f.write(classification_report(tar_te,gb_preds_te,target_names=target_names))


# Initiate, fit, and predict targets with a random forest classifier
rfcl = RandomForestClassifier(class_weight='balanced',criterion='entropy',max_features='log2',n_estimators=50,oob_score=False)
rfcl.fit(inp_tr,tar_tr.ravel())
rf_preds_tr = rfcl.predict(inp_tr)
rf_preds_te = rfcl.predict(inp_te)
f.write("\n \nRF c2d Results\n")
f.write(classification_report(tar_tr,rf_preds_tr,target_names=target_names))
f.write("\nRF NGC 2264 Results\n")
f.write(classification_report(tar_te,rf_preds_te,target_names=target_names))


# Initiate, fit, and predict targets with an eXtreme gradient boosting classifier
xgbcl = xgb.XGBClassifier(max_depth=7,sampling_method='uniform',subsample=0.5,use_label_encoder=False,eval_metric='mlogloss')
xgbcl.fit(inp_tr,tar_tr.ravel())
xgb_preds_tr = xgbcl.predict(inp_tr)
xgb_preds_te = xgbcl.predict(inp_te)
f.write("\n \nXGB c2d Results\n")
f.write(classification_report(tar_tr,xgb_preds_tr,target_names=target_names))
f.write("\nXGB NGC 2264 Results\n")
f.write(classification_report(tar_te,xgb_preds_te,target_names=target_names))
f.close()

joblib.dump(rfcl,"../CM_Settings/YSO_RF_Settings.joblib") 