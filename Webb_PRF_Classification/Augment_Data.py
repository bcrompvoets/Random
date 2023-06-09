import pandas as pd     # similar to Excel 
import numpy as np      # suitable for working with arrays
import matplotlib.pyplot as plt  # suitable for plotting
from sklearn.model_selection import train_test_split
print('Loading from repository, Done!')

date = 'May302023'

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
inp_df = pd.read_csv(f"DAOPHOT_Catalog_{date}_IR.csv").drop(['RA','DEC', 'Unnamed: 0', 'x', 'y', 'RAJ2000', 'DEJ2000',
       'mag_IR1', 'e_mag_IR1', 'mag_IR2', 'e_mag_IR2', 'mag_IR3', 'e_mag_IR3',
       'mag_IR4', 'e_mag_IR4'],axis=1)
print(inp_df.columns)
print('loading data, Done!')

cat_t= inp_df.copy().values

cat, cat_v = train_test_split(cat_t,train_size=.80,random_state=0)

# idx_0v=(np.where(cat_v[:,20]==0)[0])
# idx_1v=(np.where(cat_v[:,20]==1)[0])
# idx_2v=(np.where(((cat_v[:,20]!=1) & (cat_v[:,20]!=0)))[0])

# cat_v[idx_0v,20]=0
# cat_v[idx_1v,20]=1
# cat_v[idx_2v,20]=2

inp_test= cat_t[:,[0,2,4,6,8,10]]#1,3,5,7,9,11,13,15,17,19
err_test= cat_t[:,[1,3,5,7,9,11]]
tar_test= cat_t[:,-1]
# prob_test = cat_t[:,21]


print (len(cat_t))

idx_0=(np.where(cat_t[:,-1]==0)[0])
idx_1=(np.where(cat_t[:,-1]==1)[0])

n_sigma=2

sig_20=n_sigma*np.nanmean(cat_t[:,1])
sig_09=n_sigma*np.nanmean(cat_t[:,3])
sig_19=n_sigma*np.nanmean(cat_t[:,5])
sig_35=n_sigma*np.nanmean(cat_t[:,7])
sig_44=n_sigma*np.nanmean(cat_t[:,9])
sig_47=n_sigma*np.nanmean(cat_t[:,11])
# sig_77=n_sigma*np.nanmean(cat_t[:,13])
# sig_113=n_sigma*np.nanmean(cat_t[:,15])
# sig_128=n_sigma*np.nanmean(cat_t[:,17])
# sig_180=n_sigma*np.nanmean(cat_t[:,19])

mag= inp_test 
err= err_test 

mag_0=mag[idx_0]  
mag_1=mag[idx_1]  
err_0=err[idx_0]  
err_1=err[idx_1]  
# prob_0 = prob_test[idx_0]
# prob_1 = prob_test[idx_1]


def BST(mag,n_sample,n_sig=1, sig_20=sig_20, sig_09=sig_09, sig_19=sig_19, sig_35=sig_35, sig_44=sig_44, sig_47=sig_47):#, sig_77=sig_77, sig_113=sig_113, sig_128=sig_128, sig_180=sig_180):
    bts=np.random.choice(len(mag),n_sample,replace=True)
    # err=(np.random.randn(n_sample,10) + 1)/2
    err = np.random.default_rng().random((n_sample,6))
    mag_bts= mag[bts]
    # prob_bts = prob[bts]
    err[:,0]*=sig_20*n_sig
    err[:,1]*=sig_09*n_sig
    err[:,2]*=sig_19*n_sig
    err[:,3]*=sig_35*n_sig
    err[:,4]*=sig_44*n_sig
    err[:,5]*=sig_47*n_sig
    # err[:,6]*=sig_77*n_sig
    # err[:,7]*=sig_113*n_sig
    # err[:,8]*=sig_128*n_sig
    # err[:,9]*=sig_180*n_sig
    mag_dist= mag_bts+ err
    return mag_dist, err #, prob_bts
    
n_sample=25
n_sig = 3
inp=[]
tar=[]
err = []
prob = []

for k1 in range(int(10000/n_sample)):
    # mag0, err0, prob0 = BST(mag_0,prob_0,n_sample=n_sample,n_sig=n_sig)
    mag0, err0  = BST(mag_0, n_sample=n_sample,n_sig=n_sig)
    tar.append([0]*n_sample)
    inp.append(mag0) 
    err.append(err0) 
    # prob.append(prob0)
    # mag1, err1, prob1= BST(mag_1,prob_1,n_sample=n_sample,n_sig=n_sig)
    mag1, err1 = BST(mag_1,n_sample=n_sample,n_sig=n_sig)
    tar.append([1]*n_sample)
    inp.append(mag1)
    err.append(err1)
    # prob.append(prob1)


inp = np.array(inp)
inp=np.reshape(inp, (np.shape(inp)[0]*np.shape(inp)[1],6))
err = np.array(err)
err=np.reshape(err, (np.shape(err)[0]*np.shape(err)[1],6))
tar=np.reshape(tar,-1)
# prob = np.reshape(prob,-1)

n_arr=np.arange(len(inp))
np.random.shuffle(n_arr)
inp= inp[n_arr]
tar= tar[n_arr]
err = err[n_arr]
# prob = prob[n_arr]

out_df = pd.DataFrame(data=np.hstack((inp,err,tar.reshape(-1,1))),columns=inp_df.columns) #,prob.reshape(-1,1)

out_df.to_csv(f"Augmented_data_{date}.csv")
print(f'Augmented data has {len(out_df)}, with {len(out_df)/2} YSOs and {len(out_df)/2} contaminants')
print("Done making augmented data!")
