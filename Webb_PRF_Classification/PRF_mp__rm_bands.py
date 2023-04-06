import numpy as np
import pandas as pd
from custom_dataloader import replicate_data_single
import matplotlib.pyplot as plt
from PRF import prf

from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,classification_report,f1_score,recall_score,precision_score

import warnings
warnings.filterwarnings('ignore')

date = 'Feb172023'
webb_inp = pd.read_csv('CC_Catalog_'+date+'.csv')
val_df = pd.read_csv('CC_Catalog_'+date+'_SPICY_Preds.csv').dropna(subset=['SPICY_Class_0/1'])
# webb_inp = pd.read_csv('../Archive/Webb_PRF_Archive/CC_Catalog.csv')
# val_df = pd.read_csv('../Archive/Webb_PRF_Archive/CC_Catalog_SPICY_Preds.csv').dropna(subset=['SPICY_Class_0/1'])
train_df = pd.read_csv('Augmented_data_prob.csv')
all_inp = val_df

cont = True
amounts_te = []
# 090, 187, 200, 335, 444, 470, 770, 1130, 1280, 1800
inds = (0,2,4,6,8,10,12,14,16,18)

bands = [idx for idx in webb_inp.columns.values if (idx[:3].lower() == 'iso'.lower() and idx[10]=='v')]
bands1 = [idx for idx in webb_inp.columns.values if (idx[:3].lower() == 'iso'.lower() and idx[10]=='v')]
bands.append('SPICY_Prob')
# print(np.array(bands)[np.array(inds)])

# input_tr = input_va = train_df[bands].to_numpy()
# target_tr = target_va = train_df[['SPICY_Class_0/1']].to_numpy()
input_tr = train_df[bands].to_numpy()
target_tr = train_df[['SPICY_Class_0/1']].to_numpy()
input_va = val_df[bands].to_numpy()
target_va = val_df[['SPICY_Class_0/1']].to_numpy()

inp_tr, tar_tr = replicate_data_single(input_tr,target_tr,amounts=[len(target_tr[target_tr==0]),len(target_tr[target_tr==1])])
inp_va, tar_va = replicate_data_single(input_va,target_va,amounts=[len(target_va[target_va==0]),len(target_va[target_va==1])])

inp_te = webb_inp[np.array(bands1)].to_numpy()
def get_best_prf(inp_tr, inp_va, i1,i2,i3,i4,i5,i6,i7,i8,i9,i10):
    inds = np.array([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10])
    if i10 == 20:
        inds = np.array([i1,i2,i3,i4,i5,i6,i7,i8,i9])
    elif i10 == 30:
        inds = np.array([i1,i2,i3,i4,i5,i6,i7,i8])
    elif i10 == 40:
        inds = np.array([i1,i2,i3,i4,i5,i6])
    max_f1 = 0.5
    # for i in np.arange(0,10,2):
    seed = int(np.random.default_rng().random()*np.random.default_rng().random()*10000)

    inp_tr, prob_tr = inp_tr[:,:-1], inp_tr[:,-1]
    inp_va, prob_va = inp_va[:,:-1], inp_va[:,-1]
    dy_tr = np.array([np.array([x,1-x]) for x in prob_tr])
    
    prf_cls = prf(n_estimators=100, bootstrap=False, keep_proba=0.5)
    prf_cls.fit(X=inp_tr[:,inds], dX=inp_tr[:,inds+1], py=dy_tr)#y=tar_tr
    pred_va = prf_cls.predict(X=inp_va[:,inds],dX=inp_va[:,inds+1])
    pred_tr = prf_cls.predict(inp_tr[:,inds],inp_tr[:,inds+1])
    if (f1_score(tar_va,pred_va,average=None)[0] > max_f1):
            
        max_prf = prf_cls
        max_f1 = f1_score(tar_va,pred_va,average=None)[0]
        
    
    inp_te = webb_inp[np.array(bands1)].to_numpy()
    pred_te_max = max_prf.predict(X=inp_te[:,inds], dX=inp_te[:,inds+1]) 
    num_yso = len(pred_te_max[pred_te_max==0]) 

    return pred_te_max, num_yso, max_f1, pred_tr, pred_va

# def get_best_prf(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10):
#     inds = np.array([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10])
#     max_f1 = 0.5
#     # for i in np.arange(0,10,2):
#     seed = int(np.random.default_rng().random()*np.random.default_rng().random()*10000)
#     inp_tr, tar_tr = replicate_data_single(input_tr,target_tr,amounts=[len(target_tr[target_tr==0]),len(target_tr[target_tr==1])],seed=seed)
#     inp_va, tar_va = replicate_data_single(input_va,target_va,amounts=[len(target_va[target_va==0]),len(target_va[target_va==1])],seed=seed)

#     inp_tr, prob_tr = inp_tr[:,:-1], inp_tr[:,-1]
#     inp_va, prob_va = inp_va[:,:-1], inp_va[:,-1]
#     dy_tr = np.array([np.array([x,1-x]) for x in prob_tr])
    
#     prf_cls = prf(n_estimators=100, bootstrap=False, keep_proba=0.5)
#     prf_cls.fit(X=inp_tr[:,inds], dX=inp_tr[:,inds+1], py=dy_tr)#y=tar_tr
#     pred_va = prf_cls.predict(X=inp_va[:,inds],dX=inp_va[:,inds+1])
#     pred_tr = prf_cls.predict(inp_tr[:,inds],inp_tr[:,inds+1])
#     if (f1_score(tar_va,pred_va,average=None)[0] > max_f1):
            
#         max_prf = prf_cls
#         max_f1 = f1_score(tar_va,pred_va,average=None)[0]
        
    
#     inp_te = webb_inp[np.array(bands1)].to_numpy()
#     pred_te_max = max_prf.predict(X=inp_te[:,inds], dX=inp_te[:,inds+1]) 
#     num_yso = len(pred_te_max[pred_te_max==0]) 

#     return pred_te_max, num_yso, max_f1, pred_tr,tar_tr,tar_va,pred_va


bands = bands1

##-------------------------------------------
# Parallelize
# pred_tes, num_yso, max_f1 = get_best_prf(0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
all_inds = np.array([inp_tr, inp_va, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
print("Starting bootstrapping!")
import multiprocess as mp
import time
tic = time.perf_counter()
# iters = [all_inds] * 100
iterables = [np.array([inp_tr, inp_va, 2,4,6,8,10,12,14,16,18,20]), np.array([inp_tr, inp_va, 0,4,6,8,10,12,14,16,18,20]), np.array([inp_tr, inp_va, 0,2,6,8,10,12,14,16,18,20]),\
    np.array([inp_tr, inp_va, 0,2,4,8,10,12,14,16,18,20]), np.array([inp_tr, inp_va, 0,2,4,6,10,12,14,16,18,20]), np.array([inp_tr, inp_va, 0,2,4,6,8,12,14,16,18,20]), np.array([inp_tr, inp_va, 0,2,4,6,8,10,14,16,18,20]),\
        np.array([inp_tr, inp_va, 0,2,4,6,8,10,12,16,18,20]), np.array([inp_tr, inp_va, 0,2,4,6,8,10,12,14,18,20]), np.array([inp_tr, inp_va, 0,2,4,6,8,10,12,14,16,20]), np.array([inp_tr, inp_va, 0,4,6,8,12,14,16,18,30,30]), np.array([inp_tr, inp_va, 0,2,4,6,8,10,40,40,40,40])]
filt = ["F090W", "F187N", "F200W", "F335M", "F444W", "F470N", "F770W", "F1130W", "F1280W", "F1800W", "N bands", "MIRI bands"]
f_m = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, [2,10], [12,14,16,18]]
print("Band &  F1 \\% (tr) & F1 \\% (va) & \# YSOs\\\ \\hline")
for i, iters in enumerate(iterables):
    iters = [iters] * 100
    with mp.Pool(1) as pool:
        ans = pool.starmap(get_best_prf,iters)
    # toc = time.perf_counter()
    # print(f"Completed the bootstrapping in {(toc - tic)/60:0.2f} minutes!\n\n")

    pred_tes = list(map(list, zip(*ans)))[0]
    num_ysos = list(map(list, zip(*ans)))[1]
    max_f1 = list(map(list, zip(*ans)))[2]
    pred_trs = list(map(list,zip(*ans)))[3]
    pred_vas = list(map(list,zip(*ans)))[4]

    p_star = np.nanmean(pred_tes,axis=0)
    preds = np.zeros(len(p_star))
    preds[p_star>0.5] = 1 #If there is at least a 50% probability that the object is a star, it is labelled as a star. Otherwise it is a YSO (with 80% probability)
    p_yso = 1-p_star
    num_yso = len(preds[preds==0])

    p_star_tr = np.nanmean(pred_trs,axis=0)
    preds_tr = np.zeros(len(p_star_tr))
    preds_tr[p_star_tr>0.5] = 1 #If there is at least a 50% probability that the object is a star, it is labelled as a star. Otherwise it is a YSO (with 80% probability)


    p_star_va = np.nanmean(pred_vas,axis=0)
    preds_va = np.zeros(len(p_star_va))
    preds_va[p_star_va>0.5] = 1 #If there is at least a 50% probability that the object is a star, it is labelled as a star. Otherwise it is a YSO (with 80% probability)


    # print(np.shape(tar_trs),np.shape(np.array(pred_trs).reshape(-1,1).ravel()))
    # print(np.shape(tar_vas),np.shape(np.array(pred_vas).ravel()))
    # print(np.unique(tar_trs.reshape(1,-1).astype(int)),np.unique(pred_trs))
    # print(np.unique(tar_vas),np.unique(pred_vas))
    # p_star = np.nanmean(pred_tes,axis=0)
    # preds = np.zeros(len(p_star))
    # preds[p_star>0.5] = 1 #If there is at least a 50% probability that the object is a star, it is labelled as a star. Otherwise it is a YSO (with 80% probability)
    # p_yso = 1-p_star
    f1_tr = "%.2f" % f1_score(tar_tr,preds_tr)
    f1_va = "%.2f" % f1_score(tar_va,preds_va)
    num_mu = int(np.nanmean(num_ysos))
    num_std = int(np.nanstd(num_ysos))
    # print(f"{filt[i]} & {f1_tr} & {f1_va} & {num_yso} $^{num_yso-len(preds_va[p_star_va<0.05])}_{len(preds_va[p_star_va<0.95])-num_yso}$ mean: {num_mu} \pm {num_std} & FILL IN LATER\\")
    print(f"{filt[i]} & {f1_tr} & {f1_va} & {num_yso} $^+{abs(num_yso-len(preds[p_star<0.95]))}_-{abs(len(preds[p_star<0.05])-num_yso)}$ & FILL IN LATER\\")

toc = time.perf_counter()
print(f"Completed the bootstrapping in {(toc - tic)/60:0.2f} minutes!\n\n")
# print("Shape of predictions:",np.shape(pred_tes))
# print("Shape of num_ysos:",np.shape(num_yso))
# print("Shape of max_f1:",np.shape(max_f1))
# #-----------------------------------------

# # To determine classification, use mean of each row to determine probability of that object being a star. 
# # If the probability to be a star is less than 50%, then the object is a YSO with probability 1-mean
# # Make two columns: final class, and probability of being that class
# p_star = np.nanmean(pred_tes,axis=0)
# preds = np.zeros(len(p_star))
# preds[p_star>0.5] = 1 #If there is at least a 50% probability that the object is a star, it is labelled as a star. Otherwise it is a YSO (with 80% probability)
# p_yso = 1-p_star

# print("Number of YSOs with prob>50\%:",len(preds[p_yso>0.5]))
# # Make and save predictions/probabilities in csv
# CC_Webb_Classified = pd.DataFrame()

# CC_Webb_Classified['RA'] = webb_inp['RA']
# CC_Webb_Classified['DEC'] = webb_inp[['DEC']].values
# CC_Webb_Classified['size'] = webb_inp[['size']].values
# CC_Webb_Classified[np.array(bands)] = webb_inp[bands].values
# CC_Webb_Classified['Class'] = preds
# CC_Webb_Classified['Prob YSO'] = p_yso

# #----------------------------------------
# # Put NANs back into all_inp

# spitzer_bands = [idx for idx in all_inp.columns.values if (idx[:3].lower() == 'mag' or idx[:5].lower() == 'e_mag')]

# for i, s in enumerate(spitzer_bands):
#     if i%2 == 0:
#         s_ind = all_inp[all_inp[s] == max(all_inp[s].values)].index
#         all_inp[s].iloc[s_ind] = np.nan
#     else:
#         # s_ind = all_inp[all_inp[s] == max(all_inp[s].values)]
#         all_inp[s].iloc[s_ind] = np.nan


# #----------------------------------------
# from astropy.coordinates import match_coordinates_sky,SkyCoord
# import astropy.units as u
# # ADD SPICY PREDS
# j_sky = SkyCoord(CC_Webb_Classified.RA*u.deg,CC_Webb_Classified.DEC*u.deg)
# sp_sky = SkyCoord(all_inp.RA*u.deg, all_inp.DEC*u.deg)

# tol = max(CC_Webb_Classified['size'])

# idx, sep2d, _ = match_coordinates_sky(sp_sky, j_sky, nthneighbor=1, storekdtree='kdtree_sky')
# sep_constraint = sep2d < tol*u.deg

# print(np.count_nonzero(sep_constraint))

# j_matches = CC_Webb_Classified.iloc[idx[sep_constraint]]
# s2_matches = all_inp.iloc[sep_constraint]

# j_matches.reset_index(inplace=True)
# s2_matches.reset_index(drop=True,inplace=True)



# spitzer_bands = [idx for idx in s2_matches.columns.values if (idx[:3].lower() == 'mag' or idx[:5].lower() == 'e_mag')]
# spitzer_bands.append('SPICY_Class_0/1')
# spitzer_bands.append('SPICY')
# spitzer_bands.append('SPICY_Class')

# jwst_spitz_spicy_cat = pd.concat([j_matches,s2_matches[spitzer_bands]],axis=1)
# spicy_df_to_add = pd.DataFrame(data=jwst_spitz_spicy_cat[spitzer_bands].values,columns=spitzer_bands,index=jwst_spitz_spicy_cat['index'])
# # CC_Webb_Classified = pd.concat([CC_Webb_Classified,spicy_df_to_add],axis=1)
# df_to_add = pd.DataFrame()
# df_to_add[spitzer_bands] = [[np.nan]*len(spitzer_bands)]*len(CC_Webb_Classified)
# df_to_add.iloc[spicy_df_to_add.index] = spicy_df_to_add
# CC_Webb_Classified[spitzer_bands] = df_to_add

# CC_Webb_Classified.to_csv('CC_Webb_Predictions_Prob_'+date+'.csv')


# #----------------------------------------------------------------------------
# # Make table of Reiter, SPICY, and our own classifications

# reit = ["10:36:42.3 -58:38:04", "10:36:48.0 -58:38:19", "10:36:47.3 -58:38:10", "10:36:46.7 -58:38:05", "10:36:51.5 -58:37:54", "10:36:50.5 -58:37:52",\
#     "10:36:51.4 -58:37:48", "10:36:53.8 -58:37:48", "10:36:51.5 -58:37:10", "10:36:54.2 -58:36:26", "10:36:54.4 -58:36:18", "10:36:54.0 -58:37:20",\
#         "10:36:53.6 -58:35:20", "10:36:53.1 -58:37:37", "10:36:53.3 -58:37:54", "10:36:52.7 -58:38:05", "10:36:53.1 -58:37:08", "10:36:51.6 -58:36:58",\
#         "10:36:52.3 -58:38:09", "10:36:53.9 -58:36:29", "10:37:01.5 -58:37:51", "10:37:02.1 -58:36:58", "10:36:53.9 -58:36:32"]# End of third row is the end of MHO-only sources
# r_1 = SkyCoord(reit,unit=(u.hourangle, u.deg))

# reit_name = ['MHO1632','MHO1633','MHO1634','MHO1635','MHO1636','MHO1637','MHO1638','MHO1639, HH1221, HH1003A','MHO1640','MHO1643, HH1218','MHO1645, MHO1646','MHO1647, HH1002C','MHO1649','MHO1650',\
#     'MHO1651','MHO1652','MHO1641a','MHO1641b','HH1219','HH1223','HHc-3','HHc-4','HHc-5']

# tab_preds = open("Table_Reiter_SPICY_YSO.txt",'w')
# tab_preds.write("\citet{Kuhn2021} & \citet{Reiter2022} & Our Work \\\ \hline \n")

# j_sky = SkyCoord(CC_Webb_Classified.RA*u.deg,CC_Webb_Classified.DEC*u.deg)
# r_inds, sep2d, _ = match_coordinates_sky(r_1, j_sky, nthneighbor=1, storekdtree='kdtree_sky')
# sp_inds = CC_Webb_Classified.dropna(subset=['SPICY']).index
# _, inds_of_match = np.unique(np.r_[r_inds,sp_inds], return_index=True)
# matched_inds = np.r_[r_inds,sp_inds][np.sort(inds_of_match)]
# for i, m in enumerate(matched_inds):
#     df_tmp = CC_Webb_Classified.iloc[m]
#     if df_tmp[['Class']].values[0]==1:
#         y_or_s = 'C'
#     else:
#         y_or_s = 'YSO'
#     if m in r_inds:
#         if m in sp_inds:
#             tab_preds.write(f"{df_tmp.SPICY_Class}- SPICY {df_tmp.SPICY} & {reit_name[i]} & {y_or_s} \\\ \n")
#         else:
#             tab_preds.write(f"- & {reit_name[i]} & {y_or_s} \\\ \n")
#     else:
#         tab_preds.write(f"{df_tmp.SPICY_Class}- SPICY {df_tmp.SPICY} & - & {y_or_s} \\\ \n")

# tab_preds.close()


# #TMP - Make matches to Reiter have those RA/DEC
# CC_Webb_Classified.iloc[r_inds].RA = r_1.ra
# CC_Webb_Classified.iloc[r_inds].DEC = r_1.dec

# #----------------------------------------------------------------------------
# #----------------------------------------------------------------------------

# # Make Plots

# plt.style.use('ggplot')
# plt.rcParams['font.size'] = 12
# plt.rcParams['font.family'] = 'serif'


# #----------------------------------------------------------------------------
# # Scatter plot with hists for number of YSOs vs F1-Score
# print('Mean number of YSOs:',np.mean(num_yso), 'Median number of YSOs:', np.median(num_yso))
# print('Mean F1-Score:',np.mean(max_f1), 'Median F1-Score:', np.median(max_f1), 'Standard deviation F1-Score:', np.std(max_f1))
# print("Percent of number of objects above 100:",len(np.array(num_yso)[np.array(num_yso)>100])/len(np.array(num_yso))*100)
# fig = plt.figure(figsize=(6, 6),dpi=300)
# # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# # the size of the marginal axes and the main axes in both directions.
# # Also adjust the subplot parameters for a square plot.
# gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
#                       left=0.1, right=0.9, bottom=0.1, top=0.9,
#                       wspace=0.05, hspace=0.05)
# # Create the Axes.
# ax = fig.add_subplot(gs[1, 0])
# ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
# ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# # Draw the scatter plot and marginals.
# # scatter_hist(amounts_te, f1scores, ax, ax_histx, ax_histy)

# ax_histx.tick_params(axis="x", labelbottom=False)
# ax_histy.tick_params(axis="y", labelleft=False)
# ax.scatter(num_yso,max_f1)
# xmin,xmax = ax.get_xlim()
# ymin,ymax = ax.get_ylim()
# ax_histx.hist(num_yso,bins=np.arange(xmin,xmax,50))#
# ax_histy.hist(max_f1,bins=np.arange(ymin,ymax,0.01), orientation='horizontal')
# ax.set_xlabel('Amount of objects classified as YSOs')
# ax.set_ylabel('F1-Score of YSOs')
# # ax.set_xscale('log')

# plt.savefig("Figures/F1-Scoresvs_Num_YSOs_"+date+".png",dpi=300)

# nyso_f1s = pd.DataFrame(data={"F1-Scores": max_f1, "Num YSO": num_yso})
# nyso_f1s.to_csv("NumYSOs_F1Scores_"+date+".csv")

# #----------------------------------------
# # confusion matrix
# tar_va = CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['SPICY_Class_0/1']].values.astype(int)
# pred_va = CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['Class']].values
# ConfusionMatrixDisplay.from_predictions(tar_va,pred_va,cmap='Reds',display_labels=['YSO', 'Contaminant'],normalize='true')
# print(f1_score(tar_va,pred_va))
# print(classification_report(tar_va,pred_va))
# plt.grid(False)
# plt.savefig('Figures/CM_va_PRF_SPICY_'+date+'.png',dpi=300,facecolor=fig.get_facecolor())


# #----------------------------------------
# # JWST field image
# from astropy.io import fits
# from astropy.wcs import WCS


# # Plot image
# filter = "f200w"
# image_file = f"../../ngc3324/FITS/JWST_{filter}.fits"


# h = fits.getheader(image_file)
# f = fits.open(image_file)
# wcs = WCS(f[1].header)
# fig, ax = plt.subplots(figsize=(14,8),dpi=300)
# ax = plt.subplot(projection=wcs)
# plt.grid(color='white', ls='solid')
# plt.imshow(f[1].data,cmap='gray_r',vmin=0,vmax=20,origin='lower')
# ymax, ymin = ax.get_ylim()
# xmax, xmin = ax.get_xlim()


# ra_1 = r_1.ra
# dec_1 = r_1.dec

# ra_spicy = CC_Webb_Classified.RA.values[CC_Webb_Classified['SPICY_Class_0/1']==0]
# dec_spicy = CC_Webb_Classified.DEC.values[CC_Webb_Classified['SPICY_Class_0/1']==0]


# ra_yso = CC_Webb_Classified.RA.values[CC_Webb_Classified.Class == 0]
# dec_yso = CC_Webb_Classified.DEC.values[CC_Webb_Classified.Class == 0]

# plt.scatter(ra_yso,dec_yso, marker='*', s=150,alpha=0.4,transform=ax.get_transform('fk5'),label='Our YSOs')
# plt.scatter(ra_spicy,dec_spicy, marker='s',s=150, alpha=0.4,transform=ax.get_transform('fk5'),label='SPICY YSOs')
# plt.scatter(ra_1,dec_1, marker='o',s=150, alpha=0.4,transform=ax.get_transform('fk5'),label='Reiter et al. 2022 YSOs')
# ax.set_ylim(ymax, ymin)
# ax.set_xlim(xmax, xmin)
# plt.legend(loc=1)
# ax.grid(False)
# # plt.xticks()
# # plt.yticks()
# # plt.xlabel('RA')
# # plt.ylabel('DEC')

# plt.savefig(f"CC_w_Reiter_SPICYtr_Label_{filter}_"+date+".png",dpi=300)


# #--------------------------------------------------------------------
# # Compare SEDs and number of bands available for validation set. 
# a = CC_Webb_Classified[CC_Webb_Classified.Class==0][CC_Webb_Classified['SPICY_Class_0/1']==0].index # Correctly classified as YSO
# b = CC_Webb_Classified[CC_Webb_Classified.Class==0][CC_Webb_Classified['SPICY_Class_0/1']==1].index # Incorrectly classified as YSO
# c = CC_Webb_Classified[CC_Webb_Classified.Class!=0][CC_Webb_Classified['SPICY_Class_0/1']==0].index # Incorrectly classified as Star
# d = CC_Webb_Classified[CC_Webb_Classified.Class!=0][CC_Webb_Classified['SPICY_Class_0/1']==1].index # Correctly classified as Star
# diffs = all_inp.mag_IR2-all_inp.isophotal_vegamag_f444w

# def sed_plot_mu(ax, ind, cat,title=None,correction=0):
#     mu = pd.DataFrame([cat.iloc[ind].mean(skipna=True)])
#     sig = pd.DataFrame([cat.iloc[ind].std(skipna=True)])

#     plt.rcParams['font.size'] = 12
#     plt.rcParams['font.family'] = 'serif'
#     plt.style.use('ggplot')
#     plt.gca().invert_yaxis()

#     kwargs = {
#         'marker': 'o',
#         # 'linestyle': '-.',
#         'alpha': 0.2
#     }

#     webb_bands = [idx for idx in mu.columns.values if (idx[:14].lower() == 'isophotal_vega' and (idx[-9:-6] != 'err' and idx[-15:-12] != 'err'and idx[-10:-7] != 'err'))]
#     webb_mic = [int(webb_bands[i].split('f')[-1][:-1])/100 for i in np.arange(0,len(webb_bands))]

#     spitz2m_bands = [idx for idx in mu.columns.values if (idx[:3].lower() == 'mag')]
#     spitz_mic = [1.235,1.662,2.159,3.6,4.5,5.8,8.0]

#     all_mic = list(np.r_[webb_mic,spitz_mic]) # Collect list of values for xticks
#     del all_mic[4] 
#     del all_mic[2]
#     del all_mic[1]# Delete the xtick label of 4.44 to avoid over crowding

#     ax.plot(np.array([webb_mic]*len(cat.iloc[ind])).transpose(),(cat.iloc[ind][webb_bands].to_numpy()+correction).transpose(),'--',c='r',alpha=0.7)
#     ax.plot(webb_mic,mu[webb_bands].to_numpy()[0]+correction,**kwargs,c='r',label='Webb SED')
#     ax.fill_between(webb_mic,mu[webb_bands].to_numpy()[0]+correction-sig[webb_bands].to_numpy()[0],mu[webb_bands].to_numpy()[0]+correction+sig[webb_bands].to_numpy()[0],color='r',alpha=0.1)
#     ax.plot(np.array([spitz_mic]*len(cat.iloc[ind])).transpose(),(cat.iloc[ind][spitz2m_bands].to_numpy()+correction).transpose(),'--',c='b',alpha=0.5)
#     ax.plot(spitz_mic,mu[spitz2m_bands].to_numpy()[0],**kwargs, c='b',label='Spitzer/2MASS SED')
#     ax.fill_between(spitz_mic,mu[spitz2m_bands].to_numpy()[0]-sig[spitz2m_bands].to_numpy()[0],mu[spitz2m_bands].to_numpy()[0]+sig[spitz2m_bands].to_numpy()[0],color='b',alpha=0.1)
#     ax.plot([],[],alpha=0,label=f'Number: {len(ind)}')

#     ax.set_title(title,c='k')

#     return ax


# webb_bands = [idx for idx in CC_Webb_Classified.columns.values if (idx[:14].lower() == 'isophotal_vega' and (idx[-9:-6] != 'err' and idx[-15:-12] != 'err'and idx[-10:-7] != 'err'))]
    
# fig, axs = plt.subplots(4,2,figsize=(10,10),dpi=300)

# plt.tight_layout()
# fig.set_tight_layout(True)
# # subfigs = fig.subfigures(1, 2)

# axs[0][0].invert_yaxis()
# axs[1][0].invert_yaxis()
# axs[2][0].invert_yaxis()
# axs[3][0].invert_yaxis()
# # axs[3][1].invert_yaxis()
# axs[0][0] = sed_plot_mu(axs[0][0],a,CC_Webb_Classified,title='Consistently Classified YSOs',correction=np.nanmean(diffs))
# if len(b) != 0:
#     axs[1][0] = sed_plot_mu(axs[1][0],b,CC_Webb_Classified,title='Contaminants Classified as YSO',correction=np.nanmean(diffs))
# if len(c) != 0:
#     axs[2][0] = sed_plot_mu(axs[2][0],c,CC_Webb_Classified,title='YSOs Classified as Contaminants',correction=np.nanmean(diffs))
# axs[3][0] = sed_plot_mu(axs[3][0],d,CC_Webb_Classified,title='Consistently Classified Contaminants',correction=np.nanmean(diffs))

# ylim_a = axs[0][0].get_ylim()[1]
# ylim_b = axs[1][0].get_ylim()[1]
# ylim_c = axs[2][0].get_ylim()[1]
# ylim_d = axs[3][0].get_ylim()[1]
# axs[0][0].text(0.25, ylim_a+1, 'A',  fontsize=16, fontweight='bold', va='top',c='k')
# axs[1][0].text(0.25, ylim_b+1, 'B',  fontsize=16, fontweight='bold', va='top',c='k')
# axs[2][0].text(0.25, ylim_c+1, 'C',  fontsize=16, fontweight='bold', va='top',c='k')
# axs[3][0].text(0.25, ylim_d+1, 'D',  fontsize=16, fontweight='bold', va='top',c='k')


# axs[0][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
# axs[1][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
# axs[2][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
# axs[3][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
# plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
# axs[3][0].set_xlabel('Wavelength')
# axs[1][0].set_ylabel('Magnitude (Vega)')


# axs[0][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in a],bins=np.arange(1,11,1))
# axs[1][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in b],bins=np.arange(1,11,1))
# axs[2][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in c],bins=np.arange(1,11,1))
# axs[3][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in d],bins=np.arange(1,11,1))
# axs[2][1].set_xlim(0,11)
# axs[3][1].set_xlabel('Bands Available')

# plt.savefig('Figures/seds_'+date+'.png',dpi=300)

# #--------------------------------------------------------------------
# # Plot of Prob YSO vs recall/precision
# tar_va = CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['SPICY_Class_0/1']].values.astype(int)
# prob_yso_SP = CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['Prob YSO']].values
# prob_yso = CC_Webb_Classified[['Prob YSO']].values

# # f1s = []
# # recs = []
# # pres = []
# # cuts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1]
# # for i in cuts:
# #     preds = np.array([1]*len(CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['Class']].values))
# #     preds[np.where(prob_yso>i)[0]] = 0
# #     f1s.append(f1_score(tar_va,preds,average=None)[0])
# #     recs.append(recall_score(tar_va,preds,average=None)[0])
# #     pres.append(precision_score(tar_va,preds,average=None)[0])

# # fig, ax = plt.subplots()
# # ax.plot(cuts,f1s,label='F1-Score')
# # ax.plot(cuts,pres,label='Precision')
# # ax.plot(cuts,recs,label='Recall')
# # ax.set_xlabel('Probability YSO Cut')
# # ax.set_ylabel('Metric Score')

# # plt.legend()

# # plt.savefig('Prob_YSO_vs_metric_'+date+'.png',dpi=300)

# # tar_va = CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['SPICY_Class_0/1']].values.astype(int)
# # prob_yso_SP = CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['Prob YSO']].values
# # prob_yso = CC_Webb_Classified['Prob YSO'].values

# f1s = []
# recs = []
# pres = []
# nums = []
# cuts = np.arange(0.0,1.05,0.05)
# for i in cuts:
#     preds = np.array([1]*len(CC_Webb_Classified.dropna(subset='SPICY_Class_0/1')[['Class']].values))
#     preds[np.where(prob_yso_SP>i)[0]] = 0
#     f1s.append(f1_score(tar_va,preds,average=None)[0])
#     recs.append(recall_score(tar_va,preds,average=None)[0])
#     pres.append(precision_score(tar_va,preds,average=None)[0])
#     preds = np.array([1]*len(CC_Webb_Classified['Class'].values))
#     preds[np.where(prob_yso>i)[0]] = 0
#     nums.append(len(preds[preds==0]))

# fig, ax = plt.subplots(dpi=300)
# a1, = ax.plot(cuts,f1s,label='F1-Score')
# a2, = ax.plot(cuts,pres,label='Precision')
# a3, = ax.plot(cuts,recs,label='Recall')
# ax.set_xlabel('Probability YSO Cut')
# ax.set_ylabel('Metric Score')
# ax.set_xticks(np.arange(0,1.1,0.1))

# ax2 = ax.twinx()
# a4, = ax2.plot(cuts[:-1],nums[:-1],'#777777',label='Number YSOs')
# ax2.set_ylabel('Number of YSOs')
# ax2.grid(False)  
# ax2.set_yscale('log')

# lns = [a1, a2, a3, a4]
# ax2.legend(handles=lns, loc='lower left')
# plt.savefig('Figures/Prob_YSO_vs_metric_'+date+'.png',dpi=300)



# #--------------------------------------------------------------------
# # Histogram of brightnesses with wwavelength

# fig, axs = plt.subplots(5,2,figsize=(6,8),dpi=300)
# plt.tight_layout()
# bins =np.arange(4,20,2)
# axs[0][0].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f090w,bins=bins,label='F090W')
# axs[0][1].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f187n,bins=bins,label='F187N')
# axs[1][0].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f200w,bins=bins,label='F200W')
# axs[1][1].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f335m,bins=bins,label='F335M')
# axs[2][0].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f444w,bins=bins,label='F444W')
# axs[2][1].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0]['isophotal_vegamag_f444w-f470n'],bins=bins,label='F444W-F470N')
# axs[3][0].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f770w,bins=bins,label='F770W')
# axs[3][1].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f1130w,bins=bins,label='F1130W')
# axs[4][0].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f1280w,bins=bins,label='F1280W')
# axs[4][1].hist(CC_Webb_Classified[CC_Webb_Classified.Class==0].isophotal_vegamag_f1800w,bins=bins,label='F1800W')

# axs[0][0].set_title('F090W')
# axs[0][1].set_title('F187N')
# axs[1][0].set_title('F200W')
# axs[1][1].set_title('F335M')
# axs[2][0].set_title('F444W')
# axs[2][1].set_title('F444W-F470N')
# axs[3][0].set_title('F770W')
# axs[3][1].set_title('F1130W')
# axs[4][0].set_title('F1280W')
# axs[4][1].set_title('F1800W')
# plt.savefig("Figures/Brightness_Band_YSO_"+date+".png",dpi=300)
