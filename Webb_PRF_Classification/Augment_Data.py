import pandas as pd     # similar to Excel 
import numpy as np      # suitable for working with arrays
import matplotlib.pyplot as plt  # suitable for plotting
from sklearn.model_selection import train_test_split
print('Loading from repository, Done!')

date = 'July242023'
num_objs = 10000 # Number of objects wanted from each class to end.

cols = ['f090w', 'e_f090w', 'f187n', 'e_f187n',
       'f200w', 'e_f200w', 'f335m', 'e_f335m', 'f444w', 'e_f444w', 'f470n','e_f470n', 
       'Prob','Class']
inp_df = pd.read_csv(f"DAOPHOT_Catalog_{date}_IR.csv").dropna(subset=cols)#['f090w','f200w','f335m','f444w','f470n'])
# inp_df = pd.read_csv(f"Test_Delta_fitted_class.csv").dropna(subset=cols)
print('loading data, Done!')

inp_df, val_df = train_test_split(inp_df,train_size=.9,random_state=0)
inp_df = inp_df[cols].copy()
cat_t= inp_df.values
cat_v= val_df[cols].copy().values
print("Augmented data will be made based on ", len(cat_t[cat_t[:,-1]==0]), "YSOs and ",len(cat_t[cat_t[:,-1]==1])," contaminants.")
print("This leaves ", len(cat_v[cat_v[:,-1]==0]), "YSOs and ",len(cat_v[cat_v[:,-1]==1])," contaminants for a validation set.")
val_df.to_csv(f"Validation_data_prob_{date}.csv")



inds = np.arange(0,len(inp_df.columns)-2,2)

inp_test= cat_t[:,inds]#1,3,5,7,9,11,13,15,17,19
err_test= cat_t[:,inds+1]
tar_test= cat_t[:,-1]
prob_test = cat_t[:,-2]



idx_0=(np.where(tar_test==0)[0])
idx_1=(np.where(tar_test==1)[0])
mag= inp_test 
err= err_test 
idx_0_v =(np.where(cat_v[:,-1]==0)[0])
idx_1_v =(np.where(cat_v[:,-1]==1)[0])
mag_v = cat_v[:,inds] 
err_v = cat_v[:,inds+1] 

# print (len(idx_0), len(idx_1))

mag_0=mag[idx_0]  
mag_1=mag[idx_1]  
err_0=err[idx_0]  
err_1=err[idx_1]  
prob_0 = prob_test[idx_0]
prob_1 = prob_test[idx_1]

n_sigma=1

sig = []
sig_v = []
for i in inds+1:
    sig.append(n_sigma*np.nanmean(cat_t[:,i]))
    sig_v.append(n_sigma*np.nanmean(cat_v[:,i]))

# print(np.shape(sig))
def BST(mag,prob, sig, n_sample, n_sig=1):
    bts=np.random.choice(len(mag),n_sample,replace=True)
    err = np.random.default_rng().random((n_sample,np.shape(sig)[0]))
    sign = np.random.default_rng().choice([-1,1],(n_sample,np.shape(sig)[0]))
    mag_bts= mag[bts]
    prob_bts = prob[bts]
    err = abs(err*sig*n_sig*sign)
    mag_dist= mag_bts+ err
    return mag_dist, err, prob_bts
    
n_sample=25
n_sig = 2
inp=[]
tar=[]
err = []
prob = []

inp_v=[]
tar_v=[]
err_v = []
prob_v = []

for k1 in range(int(num_objs*0.9/n_sample)):
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


for k1 in range(int((num_objs*0.1)/n_sample)):
    mag0_v, err0_v, prob0_v = BST(mag_v[idx_0_v],cat_v[idx_0_v,-2],sig=sig_v,n_sample=n_sample,n_sig=n_sig)
    tar_v.append([0]*n_sample)
    inp_v.append(mag0_v) 
    err_v.append(err0_v) 
    prob_v.append(prob0_v)
    mag1_v, err1_v, prob1_v = BST(mag_v[idx_1_v],cat_v[idx_1_v,-2],sig=sig_v,n_sample=n_sample,n_sig=n_sig)
    tar_v.append([1]*n_sample)
    inp_v.append(mag1_v) 
    err_v.append(err1_v) 
    prob_v.append(prob1_v)
    

inp = np.array(inp)
inp=np.reshape(inp, (np.shape(inp)[0]*np.shape(inp)[1],len(inds)))
err = np.array(err)
err=np.reshape(err, (np.shape(err)[0]*np.shape(err)[1],len(inds)))
tar=np.reshape(tar,-1)
prob = np.reshape(prob,-1)

inp_v = np.array(inp_v)
inp_v = np.reshape(inp_v, (np.shape(inp_v)[0]*np.shape(inp_v)[1],len(inds)))
err_v = np.array(err_v)
err_v = np.reshape(err_v, (np.shape(err_v)[0]*np.shape(err_v)[1],len(inds)))
tar_v = np.reshape(tar_v,-1)
prob_v = np.reshape(prob_v,-1)
train_v = np.array([False]*len(inp_v))

n_arr=np.arange(len(inp))
np.random.shuffle(n_arr)
inp= inp[n_arr]
tar= tar[n_arr]
err = err[n_arr]
prob = prob[n_arr]
train = np.array([True]*len(inp))

t_df = pd.DataFrame(data=np.hstack((inp,err,prob.reshape(-1,1),tar.reshape(-1,1),train.reshape(-1,1))),columns=[
    'f090w',  'f187n', 'f200w', 'f335m',  'f444w', 'f470n',
    'e_f090w','e_f187n','e_f200w', 'e_f335m', 'e_f444w','e_f470n',
    'Prob','Class','Train'])

v_df = pd.DataFrame(data=np.hstack((inp_v,err_v,prob_v.reshape(-1,1),tar_v.reshape(-1,1),train_v.reshape(-1,1))),columns=[
    'f090w',  'f187n', 'f200w', 'f335m',  'f444w', 'f470n',
    'e_f090w','e_f187n','e_f200w', 'e_f335m', 'e_f444w','e_f470n',
    'Prob','Class','Train'])
out_df = pd.concat([t_df,v_df])

# # Add in colours and deltas and slopes
# filt_vals = [0.9, 1.87, 2.00, 3.35, 4.44, 4.70]
# print("Adding colours and deltas in")
# dao = pd.read_csv(f'DAOPHOT_Catalog_{date}.csv')
# filters = [f for f in dao.columns if (f[0]=='f') and ('-' not in f)]
# print(filters)
# dao_tmp =dao.copy()
# dao_tmp.dropna(inplace=True)
# for filt in filters:
#     dao_tmp = dao_tmp[dao_tmp['e_'+filt]<0.05]
# out_df["f090w-f444w"] = out_df['f090w'] - out_df['f444w']
# out_df["e_f090w-f444w"] = np.sqrt(out_df['e_f090w'].values**2+out_df['e_f444w'].values**2)
# for f, filt in enumerate(filters):
#     out_df[filt+"-"+filters[f+1]] = out_df[filt] - out_df[filters[f+1]]
#     out_df["e_"+filt+"-"+filters[f+1]] = np.sqrt(out_df["e_"+filt].values**2 + out_df["e_"+filters[f+1]].values**2)
#     lin_fit = np.polyfit(dao_tmp['f090w'] - dao_tmp['f444w'], dao_tmp[filt]-dao_tmp[filters[f+1]], 1)
#     out_df["δ_"+filt+"-"+filters[f+1]] = out_df[filt]-out_df[filters[f+1]] - (lin_fit[0] * (out_df['f090w'] - out_df['f444w']) + lin_fit[1])
#     out_df["e_δ_"+filt+"-"+filters[f+1]] = np.sqrt(out_df['e_'+filt].values**2+out_df['e_'+filters[f+1]].values**2)
#     out_df['slope_'+filt+'-'+filters[f+1]] = (out_df[filt]-out_df[filters[f+1]])/(filt_vals[f]-filt_vals[f+1])
#     out_df['e_slope_'+filt+'-'+filters[f+1]] = out_df["e_"+filt+"-"+filters[f+1]]/(filt_vals[f]-filt_vals[f+1])
#     if f == len(filters)-2:
#         break

# out_df["(f090w-f200w)-(f200w-f444w)"] = out_df['f090w']-2*out_df['f200w']+out_df['f444w']
# out_df["e_(f090w-f200w)-(f200w-f444w)"] = np.sqrt(out_df['e_f090w'].values**2+2*out_df['e_f200w'].values**2+out_df['e_f444w'].values**2)
# lin_fit = np.polyfit(dao_tmp['f090w'] - dao_tmp['f444w'], dao_tmp['f090w']-2*dao_tmp['f200w']+dao_tmp['f444w'], 1)
# out_df["δ_(f090w-f200w)-(f200w-f444w)"] = out_df['f090w']-2*out_df['f200w']+out_df['f444w'] - (lin_fit[0] * (out_df['f090w'] - out_df['f444w']) + lin_fit[1])
# out_df["e_δ_(f090w-f200w)-(f200w-f444w)"] = np.sqrt(out_df['e_f090w'].values**2+2*out_df['e_f200w'].values**2+out_df['e_f444w'].values**2)


# # dao['Sum1'] = dao['δ_(f090w-f200w)-(f200w-f444w)']-dao['δ_f200w-f335m']-dao['δ_f335m-f444w']
# # dao['e_Sum1'] = np.sqrt(dao['e_δ_(f090w-f200w)-(f200w-f444w)']**2+dao['e_δ_f200w-f335m']**2+dao['e_δ_f335m-f444w']**2)
# out_df['Sum1'] = out_df['δ_(f090w-f200w)-(f200w-f444w)']+out_df['δ_f090w-f187n']-out_df['δ_f200w-f335m']-out_df['δ_f335m-f444w']
# out_df['e_Sum1'] = np.sqrt(out_df['e_δ_(f090w-f200w)-(f200w-f444w)']**2+out_df['e_δ_f090w-f187n']**2+out_df['e_δ_f200w-f335m']**2+out_df['e_δ_f335m-f444w']**2)



out_df.to_csv(f"Augmented_data_prob_{date}.csv")
print(f'Augmented data has {len(out_df)}, with {len(out_df)/2} YSOs and {len(out_df)/2} contaminants')
print("Done making augmented data!")
