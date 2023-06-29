import pandas as pd     # similar to Excel 
import numpy as np      # suitable for working with arrays
import matplotlib.pyplot as plt  # suitable for plotting
from sklearn.model_selection import train_test_split
print('Loading from repository, Done!')

date = 'June192023'

# inp = pd.read_csv('catalog.csv' )
# inp = pd.read_csv('/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Webb_PRF_Classification/CC_Webb_Predictions_Prob_Feb172023_Spitz_ONLY.csv')
# inp = pd.read_csv('CC_Catalog_2_5sig'+date+'_SPICY_Preds.csv')[['isophotal_vegamag_f200w',
#        'isophotal_vegamag_err_f200w', 'isophotal_vegamag_f090w',
#        'isophotal_vegamag_err_f090w', 'isophotal_vegamag_f187n',
#        'isophotal_vegamag_err_f187n', 'isophotal_vegamag_f335m',
#        'isophotal_vegamag_err_f335m', 'isophotal_vegamag_f444w',
#        'isophotal_vegamag_err_f444w', 'isophotal_vegamag_f444w-f470n',
#        'isophotal_vegamag_err_f444w-f470n', 'isophotal_vegamag_f770w',
#        'isophotal_vegamag_err_f770w', 'isophotal_vegamag_f1130w',
#        'isophotal_vegamag_err_f1130w', 'isophotal_vegamag_f1280w',
#        'isophotal_vegamag_err_f1280w', 'isophotal_vegamag_f1800w',
#        'isophotal_vegamag_err_f1800w','SPICY_Class_0/1','SPICY_Prob']]
inp_df = pd.read_csv(f"DAOPHOT_Catalog_{date}_IR.csv")[['f090w', 'e_f090w', 'f187n', 'e_f187n',
       'f200w', 'e_f200w', 'f335m', 'e_f335m', 'f444w', 'e_f444w', 'f470n',
       'e_f470n', 'f090w-f444w', 'e_f090w-f444w', 'f090w-f187n',
       'e_f090w-f187n', 'δ_f090w-f187n', 'e_δ_f090w-f187n', 'f187n-f200w',
       'e_f187n-f200w', 'δ_f187n-f200w', 'e_δ_f187n-f200w', 'f200w-f335m',
       'e_f200w-f335m', 'δ_f200w-f335m', 'e_δ_f200w-f335m', 'f335m-f444w',
       'e_f335m-f444w', 'δ_f335m-f444w', 'e_δ_f335m-f444w', 'f444w-f470n',
       'e_f444w-f470n', 'δ_f444w-f470n', 'e_δ_f444w-f470n',
       '(f090w-f200w)-(f200w-f444w)', 'e_(f090w-f200w)-(f200w-f444w)',
       'δ_(f090w-f200w)-(f200w-f444w)', 'e_δ_(f090w-f200w)-(f200w-f444w)',
       'Sum1', 'e_Sum1','Prob','Class']].dropna()
print('loading data, Done!')

cat_t= inp_df.copy().values

# cat, cat_v = train_test_split(cat_t,train_size=.80,random_state=0)

inds = np.arange(0,len(inp_df.columns)-2,2)

inp_test= cat_t[:,inds]#1,3,5,7,9,11,13,15,17,19
err_test= cat_t[:,inds+1]
tar_test= cat_t[:,-1]
prob_test = cat_t[:,-2]



idx_0=(np.where(tar_test==0)[0])
idx_1=(np.where(tar_test==1)[0])
mag= inp_test 
err= err_test 

print (len(idx_0), len(idx_1))

mag_0=mag[idx_0]  
mag_1=mag[idx_1]  
err_0=err[idx_0]  
err_1=err[idx_1]  
prob_0 = prob_test[idx_0]
prob_1 = prob_test[idx_1]

n_sigma=1

sig = []
for i in inds+1:
    sig.append(n_sigma*np.nanmean(cat_t[:,i]))

def BST(mag,prob, sig, n_sample, n_sig=1):
    bts=np.random.choice(len(mag),n_sample,replace=True)
    err = np.random.default_rng().random((n_sample,20))
    mag_bts= mag[bts]
    prob_bts = prob[bts]
    err = err*sig*n_sig
    mag_dist= mag_bts+ err
    return mag_dist, err, prob_bts
    
n_sample=25
n_sig = 2
inp=[]
tar=[]
err = []
prob = []

for k1 in range(int(10000/n_sample)):
    mag0, err0, prob0 = BST(mag_0,prob_0,sig=sig,n_sample=n_sample,n_sig=n_sig)
    tar.append([0]*n_sample)
    inp.append(mag0) 
    err.append(err0) 
    prob.append(prob0)
    mag1, err1, prob1= BST(mag_1,prob_1,sig=sig,n_sample=n_sample,n_sig=n_sig)
    tar.append([1]*n_sample)
    inp.append(mag1)
    err.append(err1)
    prob.append(prob1)


inp = np.array(inp)
inp=np.reshape(inp, (np.shape(inp)[0]*np.shape(inp)[1],len(inds)))
err = np.array(err)
err=np.reshape(err, (np.shape(err)[0]*np.shape(err)[1],len(inds)))
tar=np.reshape(tar,-1)
prob = np.reshape(prob,-1)

n_arr=np.arange(len(inp))
np.random.shuffle(n_arr)
inp= inp[n_arr]
tar= tar[n_arr]
err = err[n_arr]
prob = prob[n_arr]

out_df = pd.DataFrame(data=np.hstack((inp,err,prob.reshape(-1,1),tar.reshape(-1,1))),columns=['f090w',  'f187n', 
       'f200w', 'f335m',  'f444w', 'f470n',
        'f090w-f444w','f090w-f187n', 'δ_f090w-f187n', 'f187n-f200w',
        'δ_f187n-f200w', 'f200w-f335m','δ_f200w-f335m','f335m-f444w',
        'δ_f335m-f444w', 'f444w-f470n', 'δ_f444w-f470n',
       '(f090w-f200w)-(f200w-f444w)', 'δ_(f090w-f200w)-(f200w-f444w)',
       'Sum1', 'e_f090w','e_f187n','e_f200w', 'e_f335m', 'e_f444w','e_f470n',
        'e_f090w-f444w', 'e_f090w-f187n', 'e_δ_f090w-f187n', 'e_f187n-f200w', 'e_δ_f187n-f200w','e_f200w-f335m', 
        'e_δ_f200w-f335m', 'e_f335m-f444w', 'e_δ_f335m-f444w', 
       'e_f444w-f470n', 'e_δ_f444w-f470n','e_(f090w-f200w)-(f200w-f444w)', 'e_δ_(f090w-f200w)-(f200w-f444w)',
       'e_Sum1','Prob','Class'])

out_df.to_csv(f"Augmented_data_prob_{date}.csv")
print(f'Augmented data has {len(out_df)}, with {len(out_df)/2} YSOs and {len(out_df)/2} contaminants')
print("Done making augmented data!")
