import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,classification_report,f1_score,recall_score,precision_score

from astropy.coordinates import match_coordinates_sky,SkyCoord, Angle
import astropy.units as u

import warnings
warnings.filterwarnings('ignore')

date = 'June192023'
CC_Webb_Classified = pd.read_csv(f'CC_Classified_DAOPHOT_{date}.csv')#f'CC_Classified_DAOPHOT_{date}.csv'
dao_IR = pd.read_csv(f'DAOPHOT_Catalog_{date}_IR.csv')

date = 'DAOPHOT_'+ date

fcd_columns = [c for c in dao_IR.columns if c[0] == "f" or c[0]=='δ'or c[:3]=='Sum']
errs = ["e_"+f for f in fcd_columns]
bands = fcd_columns+errs

# ----------------------------------------------------------------------------
# Print classification reports
print("RF Classification Report")

print(classification_report(CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns).Init_Class,CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns).Class_RF))
print(classification_report(CC_Webb_Classified.dropna(subset='Init_Class').Init_Class,CC_Webb_Classified.dropna(subset='Init_Class').Class_PRF))

# ----------------------------------------------------------------------------
# Make table of Reiter, SPICY, and our own classifications

reit = ["10:36:42.3 -58:38:04", "10:36:48.0 -58:38:19", "10:36:47.3 -58:38:10", "10:36:46.7 -58:38:05", "10:36:51.5 -58:37:54", "10:36:50.5 -58:37:52",\
    "10:36:51.4 -58:37:48", "10:36:53.8 -58:37:48", "10:36:51.5 -58:37:10", "10:36:54.2 -58:36:26", "10:36:54.4 -58:36:18", "10:36:54.0 -58:37:20",\
        "10:36:53.6 -58:35:20", "10:36:53.1 -58:37:37", "10:36:53.3 -58:37:54", "10:36:52.7 -58:38:05", "10:36:53.1 -58:37:08", "10:36:51.6 -58:36:58",\
        "10:36:52.3 -58:38:09", "10:36:53.9 -58:36:29", "10:37:01.5 -58:37:51", "10:37:02.1 -58:36:58", "10:36:53.9 -58:36:32"]# End of third row is the end of MHO-only sources
r_1 = SkyCoord(reit,unit=(u.hourangle, u.deg))

reit_name = ['MHO1632','MHO1633','MHO1634','MHO1635','MHO1636','MHO1637','MHO1638','MHO1639, HH1221, HH1003A','MHO1640','MHO1643, HH1218','MHO1645, MHO1646','MHO1647, HH1002C','MHO1649','MHO1650',\
    'MHO1651','MHO1652','MHO1641a','MHO1641b','HH1219','HH1223','HHc-3','HHc-4','HHc-5']

tab_preds = open("Table_Reiter_SPICY_YSO.txt",'w')
tab_preds.write("\citet{Kuhn2021} & \citet{Reiter2022} & Our Work \\\ \hline \n")

r_inds, sep2d, _ = match_coordinates_sky(r_1, SkyCoord(CC_Webb_Classified.RA,CC_Webb_Classified.DEC,unit=u.deg), nthneighbor=1, storekdtree='kdtree_sky')
sp_inds = CC_Webb_Classified[CC_Webb_Classified.Init_Class==0].index
_, inds_of_match = np.unique(np.r_[r_inds,sp_inds], return_index=True)
matched_inds = np.r_[r_inds,sp_inds][np.sort(inds_of_match)]

for i, m in enumerate(matched_inds):
    df_tmp = CC_Webb_Classified.iloc[m]
    # df_tmp_ir = dao_IR.iloc[i]
    if df_tmp[['Class_RF']].values[0]==0 and df_tmp[['Class_PRF']].values[0]==0:
        y_or_s = 'YSO - PRF and RF'
    elif df_tmp[['Class_RF']].values[0]==0:
        y_or_s = 'YSO - RF'
    elif df_tmp[['Class_PRF']].values[0]==0:
        y_or_s = 'YSO - PRF'
    else:
        y_or_s = 'C'
    if m in r_inds:
        if m in sp_inds:
            tab_preds.write(f"{df_tmp.Survey} - SPICY {df_tmp.SPICY_ID} & {reit_name[i]} & {y_or_s} \\\ \n")
        else:
            tab_preds.write(f"- & {reit_name[i]} & {y_or_s} \\\ \n")
    else:
        tab_preds.write(f"{df_tmp.Survey} - SPICY {df_tmp.SPICY_ID} & - & {y_or_s} \\\ \n")

tab_preds.close()


# #TMP - Make matches to Reiter have those RA/DEC
# CC_Webb_Classified.iloc[r_inds].RA = r_1.ra
# CC_Webb_Classified.iloc[r_inds].DEC = r_1.dec

# #----------------------------------------------------------------------------
# #----------------------------------------------------------------------------

# Make Plots

plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'


# #----------------------------------------------------------------------------
# Scatter plot with hists for number of YSOs vs F1-Score
num_yso_rf = np.loadtxt("Data/Num_YSOs_RFDAOPHOT_June192023")
num_yso_prf = np.loadtxt("Data/Num_YSOs_PRFDAOPHOT_June192023")
max_f1_rf = np.loadtxt("Data/Max_f1s_RFDAOPHOT_June192023")
max_f1_prf = np.loadtxt("Data/Max_f1s_PRFDAOPHOT_June192023")

# print('Mean number of YSOs:',np.mean(num_yso), 'Median number of YSOs:', np.median(num_yso))
# print('Mean F1-Score:',np.mean(max_f1), 'Median F1-Score:', np.median(max_f1), 'Standard deviation F1-Score:', np.std(max_f1))
fig = plt.figure(figsize=(6, 6),dpi=300)
fig.tight_layout()
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
# scatter_hist(amounts_te, f1scores, ax, ax_histx, ax_histy)

ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
ax.scatter(num_yso_rf,max_f1_rf,label='RF')
ax.scatter(num_yso_prf,max_f1_prf,label='PRF')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ax_histx.hist(num_yso_rf,bins=np.arange(xmin,xmax,5),histtype='step')#
ax_histx.hist(num_yso_prf,bins=np.arange(xmin,xmax,5),histtype='step')#
ax_histy.hist(max_f1_rf,bins=np.arange(ymin,ymax,0.005), orientation='horizontal',histtype='step')
ax_histy.hist(max_f1_prf,bins=np.arange(ymin,ymax,0.005), orientation='horizontal',histtype='step')
ax.set_xlabel('Amount of objects classified as YSOs')
ax.set_ylabel('F1-Score of YSOs')
# ax.set_xscale('log')
ax.legend(loc='upper right')
plt.savefig("Figures/F1-Scoresvs_Num_YSOs_"+date+".png",dpi=300)


# #----------------------------------------
# # Just hist

# fig, ax = plt.subplots(figsize=(10,10),dpi=300)
# ax.hist(num_yso,bins=np.arange(xmin,xmax,50))
# ax.set_xlabel('Amount of objects classified as YSOs')
# plt.savefig("Figures/Num_YSOs_"+date+".png",dpi=300)


# #----------------------------------------
# # confusion matrix


tar_va = CC_Webb_Classified.dropna(subset='Init_Class').dropna(subset=fcd_columns)[['Init_Class']].values.astype(int)
pred_va = CC_Webb_Classified.dropna(subset='Init_Class').dropna(subset=fcd_columns)[['Class_RF']].values
ConfusionMatrixDisplay.from_predictions(tar_va,pred_va,cmap='Reds',display_labels=['YSO', 'Contaminant'],normalize='true')
# print(f1_score(tar_va,pred_va))
# print(classification_report(tar_va,pred_va))
plt.grid(False)
plt.savefig('Figures/CM_va_RF_SPICY_'+date+'.png',dpi=300,facecolor=fig.get_facecolor())
plt.close()

tar_va = CC_Webb_Classified.dropna(subset='Init_Class')[['Init_Class']].values.astype(int)
pred_va = CC_Webb_Classified.dropna(subset='Init_Class')[['Class_PRF']].values
ConfusionMatrixDisplay.from_predictions(tar_va,pred_va,cmap='Reds',display_labels=['YSO', 'Contaminant'],normalize='true')
# print(f1_score(tar_va,pred_va))
# print(classification_report(tar_va,pred_va))
plt.grid(False)
plt.savefig('Figures/CM_va_PRF_SPICY_'+date+'.png',dpi=300,facecolor=fig.get_facecolor())
plt.close()
#----------------------------------------
# JWST field image
from astropy.io import fits
from astropy.wcs import WCS


# Plot image
filter = "f090w"
f = fits.open(f'/users/breannacrompvoets/DAOPHOT/NGC3324/f090w_ADU.fits')
wcs = WCS(f[0].header)
fig, ax = plt.subplots(figsize=(14,8),dpi=300)
ax = plt.subplot(projection=wcs)
plt.grid(color='white', ls='solid')
plt.imshow(f[0].data,cmap='gray_r',vmax=1300,origin='lower')
ymax, ymin = ax.get_ylim()
xmax, xmin = ax.get_xlim()

ra_1 = r_1.ra
dec_1 = r_1.dec

ra_ir = CC_Webb_Classified.RA.values[CC_Webb_Classified['Init_Class']==0]
dec_ir = CC_Webb_Classified.DEC.values[CC_Webb_Classified['Init_Class']==0]


ra_yso_rf = CC_Webb_Classified.RA.values[CC_Webb_Classified.Class_RF == 0]
dec_yso_rf = CC_Webb_Classified.DEC.values[CC_Webb_Classified.Class_RF == 0]
ra_yso_prf = CC_Webb_Classified.RA.values[CC_Webb_Classified.Class_PRF == 0]
dec_yso_prf = CC_Webb_Classified.DEC.values[CC_Webb_Classified.Class_PRF == 0]

plt.scatter(ra_yso_rf,dec_yso_rf, marker='*', s=150,alpha=0.8,transform=ax.get_transform('fk5'),label='Our YSOs (RF)')
plt.scatter(ra_yso_prf,dec_yso_prf, marker='*', s=150,alpha=0.8,transform=ax.get_transform('fk5'),label='Our YSOs (PRF)')
plt.scatter(ra_ir,dec_ir, marker='s',s=150, alpha=0.8,transform=ax.get_transform('fk5'),label='SPICY (2021) or Ohlendorf (2013) YSOs')
plt.scatter(ra_1,dec_1, marker='o',s=150, alpha=0.8,transform=ax.get_transform('fk5'),label='Reiter et al. 2022 YSOs')
ax.set_ylim(ymax, ymin)
ax.set_xlim(xmax, xmin)
plt.legend(loc=1)
ax.grid(False)
# plt.xticks()
# plt.yticks()
# plt.xlabel('RA')
# plt.ylabel('DEC')

plt.savefig(f"Figures/field_image_{filter}_"+date+".png",dpi=300)


#--------------------------------------------------------------------
# Compare SEDs and number of bands available for validation set. 
a = CC_Webb_Classified[CC_Webb_Classified.Class_RF==0][CC_Webb_Classified['Init_Class']==0].index # Correctly classified as YSO
b = CC_Webb_Classified[CC_Webb_Classified.Class_RF==0][CC_Webb_Classified['Init_Class']==1].index # Incorrectly classified as YSO
c = CC_Webb_Classified[CC_Webb_Classified.Class_RF!=0][CC_Webb_Classified['Init_Class']==0].index # Incorrectly classified as Star
d = CC_Webb_Classified[CC_Webb_Classified.Class_RF!=0][CC_Webb_Classified['Init_Class']==1].index # Correctly classified as Star
# diffs = dao_IR.mag_IR2-all_inp.isophotal_vegamag_f444w

def sed_plot_mu(ax, ind, cat,title=None,correction=0):
    mu = pd.DataFrame([cat.iloc[ind].mean(skipna=True)])
    sig = pd.DataFrame([cat.iloc[ind].std(skipna=True)])

    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.style.use('ggplot')
    plt.gca().invert_yaxis()

    kwargs = {
        'marker': 'o',
        # 'linestyle': '-.',
        'alpha': 0.2
    }

    webb_bands = [idx for idx in mu.columns.values if (idx[0].lower() == 'f' and len(idx) == 5)]
    webb_mic = [int(webb_bands[i].split('f')[-1][:-1])/100 for i in np.arange(0,len(webb_bands))]
    
    # spitz2m_bands = [idx for idx in mu.columns.values if (idx[:3].lower() == 'mag')]
    # spitz_mic = [1.235,1.662,2.159,3.6,4.5,5.8,8.0]

    # all_mic = list(np.r_[webb_mic,spitz_mic]) # Collect list of values for xticks
    # del all_mic[4] 
    # del all_mic[2]
    # del all_mic[1]# Delete the xtick label of 4.44 to avoid over crowding

    ax.plot(np.array([webb_mic]*len(cat.iloc[ind])).transpose(),(cat.iloc[ind][webb_bands].to_numpy()+correction).transpose(),'--',c='r',alpha=0.7)
    ax.plot(webb_mic,mu[webb_bands].to_numpy()[0]+correction,**kwargs,c='r',label='Webb SED')
    ax.fill_between(webb_mic,mu[webb_bands].to_numpy()[0]+correction-sig[webb_bands].to_numpy()[0],mu[webb_bands].to_numpy()[0]+correction+sig[webb_bands].to_numpy()[0],color='r',alpha=0.1)
    # ax.plot(np.array([spitz_mic]*len(cat.iloc[ind])).transpose(),(cat.iloc[ind][spitz2m_bands].to_numpy()+correction).transpose(),'--',c='b',alpha=0.5)
    # ax.plot(spitz_mic,mu[spitz2m_bands].to_numpy()[0],**kwargs, c='b',label='Spitzer/2MASS SED')
    # ax.fill_between(spitz_mic,mu[spitz2m_bands].to_numpy()[0]-sig[spitz2m_bands].to_numpy()[0],mu[spitz2m_bands].to_numpy()[0]+sig[spitz2m_bands].to_numpy()[0],color='b',alpha=0.1)
    ax.plot([],[],alpha=0,label=f'Number: {len(ind)}')

    ax.set_title(title,c='k')

    return ax


webb_bands = [idx for idx in CC_Webb_Classified.columns.values if (idx[0].lower() == 'f' and len(idx) == 5)]
    
fig, axs = plt.subplots(4,2,figsize=(10,10),dpi=300)

plt.tight_layout()
fig.set_tight_layout(True)
# subfigs = fig.subfigures(1, 2)

axs[0][0].invert_yaxis()
axs[1][0].invert_yaxis()
axs[2][0].invert_yaxis()
axs[3][0].invert_yaxis()
# axs[3][1].invert_yaxis()
axs[0][0] = sed_plot_mu(axs[0][0],a,CC_Webb_Classified,title='Consistently Classified YSOs')#,correction=np.nanmean(diffs))
if len(b) != 0:
    axs[1][0] = sed_plot_mu(axs[1][0],b,CC_Webb_Classified,title='Contaminants Classified as YSO')#,correction=np.nanmean(diffs))
if len(c) != 0:
    axs[2][0] = sed_plot_mu(axs[2][0],c,CC_Webb_Classified,title='YSOs Classified as Contaminants')#,correction=np.nanmean(diffs))
axs[3][0] = sed_plot_mu(axs[3][0],d,CC_Webb_Classified,title='Consistently Classified Contaminants')#,correction=np.nanmean(diffs))

ylim_a = axs[0][0].get_ylim()[1]
ylim_b = axs[1][0].get_ylim()[1]
ylim_c = axs[2][0].get_ylim()[1]
ylim_d = axs[3][0].get_ylim()[1]
axs[0][0].text(0.25, ylim_a+1, 'A',  fontsize=16, fontweight='bold', va='top',c='k')
axs[1][0].text(0.25, ylim_b+1, 'B',  fontsize=16, fontweight='bold', va='top',c='k')
axs[2][0].text(0.25, ylim_c+1, 'C',  fontsize=16, fontweight='bold', va='top',c='k')
axs[3][0].text(0.25, ylim_d+1, 'D',  fontsize=16, fontweight='bold', va='top',c='k')


axs[0][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
axs[1][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
axs[2][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
axs[3][0].legend(facecolor='darkgrey', framealpha=1,loc='lower right')
plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
axs[3][0].set_xlabel('Wavelength')
axs[1][0].set_ylabel('Magnitude (Vega)')


axs[0][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in a],bins=np.arange(1,11,1))
axs[1][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in b],bins=np.arange(1,11,1))
axs[2][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in c],bins=np.arange(1,11,1))
axs[3][1].hist([np.count_nonzero(~CC_Webb_Classified[webb_bands].iloc[i].isna()) for i in d],bins=np.arange(1,11,1))
axs[2][1].set_xlim(0,11)
axs[3][1].set_xlabel('Bands Available')

plt.savefig('Figures/seds_'+date+'.png',dpi=300)

#--------------------------------------------------------------------
# Plot of Prob YSO vs recall/precision

tar_va_rf = CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)[['Init_Class']].values.astype(int)
prob_yso_rf_ir = CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)[['Prob_RF']].values
prob_yso_rf = CC_Webb_Classified['Prob_RF'].values

tar_va_prf = CC_Webb_Classified.dropna(subset='Init_Class')[['Init_Class']].values.astype(int)
prob_yso_prf_ir = CC_Webb_Classified.dropna(subset='Init_Class')[['Prob_PRF']].values
prob_yso_prf = CC_Webb_Classified['Prob_PRF'].values

f1s_rf = []
recs_rf = []
pres_rf = []
nums_rf = []

f1s_prf = []
recs_prf = []
pres_prf = []
nums_prf = []

cuts = np.arange(0.0,1.05,0.05)
for i in cuts:
    preds = np.array([1]*len(CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)[['Class_RF']].values))
    # print(np.where(prob_yso_rf_ir>i)[0])
    preds[np.where(prob_yso_rf_ir>i)[0]] = 0
    f1s_rf.append(f1_score(tar_va_rf,preds,average=None)[0])
    recs_rf.append(recall_score(tar_va_rf,preds,average=None)[0])
    pres_rf.append(precision_score(tar_va_rf,preds,average=None)[0])
    preds = np.array([1]*len(CC_Webb_Classified['Class_RF'].values))
    preds[np.where(prob_yso_rf>i)[0]] = 0
    nums_rf.append(len(preds[preds==0]))

    preds = np.array([1]*len(CC_Webb_Classified.dropna(subset='Init_Class')[['Class_PRF']].values))
    preds[np.where(prob_yso_prf_ir>i)[0]] = 0
    f1s_prf.append(f1_score(tar_va_prf,preds,average=None)[0])
    recs_prf.append(recall_score(tar_va_prf,preds,average=None)[0])
    pres_prf.append(precision_score(tar_va_prf,preds,average=None)[0])
    preds = np.array([1]*len(CC_Webb_Classified['Class_PRF'].values))
    preds[np.where(prob_yso_prf>i)[0]] = 0
    nums_prf.append(len(preds[preds==0]))

fig, ax = plt.subplots(2,1,sharex=True,dpi=300,figsize=(5,10))

ax[0].plot(cuts,f1s_rf,c='firebrick',label='F1-Score (RF)')
ax[0].plot(cuts,pres_rf,c='orangered',label='Precision (RF)')
ax[0].plot(cuts,recs_rf,c='lightcoral',label='Recall (RF)')

ax[0].plot(cuts,f1s_prf,'--',c='firebrick',label='F1-Score (PRF)')
ax[0].plot(cuts,pres_prf,'--',c='orangered',label='Precision (PRF)')
ax[0].plot(cuts,recs_prf,'--',c='lightcoral',label='Recall (PRF)')
ax[1].set_xlabel('Probability YSO Cut')
ax[0].set_ylabel('Metric Score')
ax[1].set_xticks(np.arange(0,1.1,0.1))
ax[0].set_xticks(np.arange(0,1.1,0.1))

# ax2 = ax.twinx()
ax[1].plot(cuts[:-1],nums_rf[:-1],c='darkgrey',label='Number YSOs (RF)')
ax[1].plot(cuts[:-1],nums_prf[:-1],linestyle='--',c='darkgrey',label='Number YSOs (PRF)')
ax[1].set_ylabel('Number of YSOs')
# ax[1].grid(False)  
ax[1].set_yscale('log')

# lns = [a1, a2, a3, a4]
ax[0].legend(loc='lower left')
ax[1].legend(loc='lower left')
plt.savefig('Figures/Prob_YSO_vs_metric_'+date+'.png',dpi=300)



# --------------------------------------------------------------------
# Histogram of brightnesses with wavelength

fig, axs = plt.subplots(3,2,figsize=(6,8),dpi=300)
fig.tight_layout()
bins =np.arange(6,24,2)
f = 0
for i in range(0,3):
    for j in range(0,2):
        axs[i][j].hist(CC_Webb_Classified.loc[CC_Webb_Classified.Class_RF==0,webb_bands[f]],bins=bins,label=webb_bands[f])
        axs[i][j].set_title(webb_bands[f])
        f += 1
plt.savefig("Figures/Brightness_Band_YSO_"+date+".png",dpi=300)


#  --------------------------------------------------------------------
# CMDs and CCDs with labelled YSOs


for f in fcd_columns:
    fig = plt.subplots(dpi=300)
    err = np.sqrt(CC_Webb_Classified['e_f090w-f444w'].values**2+CC_Webb_Classified['e_'+f].values**2)
    CC_tmp = CC_Webb_Classified[err<0.2]

    plt.scatter(CC_tmp['f090w-f444w'], CC_tmp[f],c=err[err<0.2],marker='.', s=1,cmap='copper')
    plt.colorbar(label='Propogated Error')

    CC_rf = CC_tmp.copy()
    CC_rf = CC_rf[CC_rf.Class_RF==0]
    plt.scatter(CC_rf['f090w-f444w'],CC_rf[f],marker='s',s=15,c='orangered',label='RF YSOs')
    CC_prf = CC_tmp.copy()
    CC_prf = CC_prf[CC_prf.Class_PRF==0]
    plt.scatter(CC_prf['f090w-f444w'],CC_prf[f],marker='*',s=15,c='maroon', label = 'PRF YSOs')

    plt.xlabel("F090W-F444W")
    plt.ylabel(f.upper())
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f"Figures/CMD_{f}.png")
    plt.close()


# ------------------
# Spitzer-DAOPHOT comparison


from mpl_toolkits.axes_grid1 import make_axes_locatable
cc_match = dao_IR
cc_match.where(cc_match!=99.999,np.nan,inplace=True)

fig, ax = plt.subplots(2,1,dpi=300)
lims = np.arange(8.5,16)
print(len(cc_match[['mag3_6']].values))
err = (cc_match[['d3_6m']].values**2+cc_match[['e_f335m']].values**2)**0.5
c = ax[0].scatter(cc_match[['mag3_6']].values,cc_match[['f335m']].values,s=5,c=err,cmap='copper')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[0].plot(lims,lims,'k',lw=0.3)
ax[0].set_xlabel('IRAC2')
ax[0].set_ylabel('F335M')
ax[0].invert_xaxis()
ax[0].invert_yaxis()
plt.colorbar(mappable=c,cax=cax)

err = (cc_match[['d4_5m']].values**2+cc_match[['e_f444w']].values**2)**0.5
c2 = ax[1].scatter(cc_match[['mag4_5']].values,cc_match[['f444w']].values,s=5,c=err,cmap='copper')
ax[1].plot(lims,lims,'k',lw=0.3)
ax[1].set_ylim(9,16.5)
ax[1].set_xlabel('IRAC3')
ax[1].set_ylabel('F444W')
ax[1].invert_xaxis()
ax[1].invert_yaxis()
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(mappable=c2,cax=cax)

plt.savefig("Figures/spitz_dao_comp.png")