import pandas as pd
import numpy as np

date = "CC_Mar42024"
filepath = "./"+date+"/"
CC_Probs = pd.read_csv(filepath+"CC_Classified_"+date+"_with_10_percent_saved.csv").copy()

runs = [c for c in CC_Probs.columns if ('-' not in c) & (c[0]=='f')]
print(runs)
for f in runs:
    CC_tmp = pd.read_csv(filepath+"CC_Classified_"+date+"_no_"+f+"_with_10_percent_saved.csv")
    CC_Probs['P_PRF_no_'+f] = CC_tmp.Prob_PRF.values
    CC_Probs['PRF_pass_thresh_no_'+f] = 1-CC_tmp.Class_PRF.values
    CC_Probs['P_RF_no_'+f] = CC_tmp.Prob_RF.values
    CC_Probs['RF_pass_thresh_no_'+f] = 1-CC_tmp.Class_RF.values
    print(f, len(CC_tmp[CC_tmp.Class_PRF==0]))

CC_Probs['PRF_frac_runs_yso'] = np.zeros(len(CC_Probs))
CC_Probs['RF_frac_runs_yso'] = np.zeros(len(CC_Probs))

for f in runs:
    CC_Probs['PRF_frac_runs_yso']=CC_Probs['PRF_frac_runs_yso']+CC_Probs['PRF_pass_thresh_no_'+f]
    CC_Probs['RF_frac_runs_yso']=CC_Probs['RF_frac_runs_yso']+CC_Probs['RF_pass_thresh_no_'+f]

CC_Probs['PRF_frac_runs_yso'] = CC_Probs['PRF_frac_runs_yso']/len(runs)
CC_Probs['RF_frac_runs_yso'] = CC_Probs['RF_frac_runs_yso']/len(runs)


# Set the class such that 0 = YSO, 1 = Contaminant. With the boolean multiplier need
# these to be opposite and then fixed: hence 1-((1-prob)*bool)
CC_Probs['Class_PRF'] = 1-((1-CC_Probs.Class_PRF.values)*(CC_Probs.PRF_frac_runs_yso.values>0.5))
CC_Probs['Class_RF'] = 1-((1-CC_Probs.Class_RF.values)*(CC_Probs.RF_frac_runs_yso.values>0.5))

CC_Probs.to_csv(filepath+"Classified_"+date+"_final_class_with_10_percent_saved.csv")

print("After masking to only those objects identified as a YSO in at least 4 out of 6 bands, we have ",
      len(CC_Probs[CC_Probs.Class_PRF==0])," cYSOs with the PRF and ", len(CC_Probs[CC_Probs.Class_RF==0]), " with the RF.")


ranked_features = pd.read_csv(filepath+'CC_feature_importances_'+date+'_with_10_percent_saved.csv',index_col=0)
for f in runs:
    rank_tmp = pd.read_csv(filepath+"CC_feature_importances_"+date+"_no_"+f+"_with_10_percent_saved.csv",index_col=0)
    for i in ranked_features.index:
        if i in rank_tmp.Feature.values:
            ranked_features.loc[i,'Importance'] = ranked_features.loc[i,'Importance']+rank_tmp.loc[i,'Importance']

ranked_features.sort_values(by='Importance',inplace=True,ascending=True)
ranked_features.to_csv(filepath+"CC_feature_importances_"+date+"_final_class_with_10_percent_saved.csv")