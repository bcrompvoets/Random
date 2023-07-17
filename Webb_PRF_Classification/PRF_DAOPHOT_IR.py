import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib as mpl

from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,classification_report,f1_score,recall_score,precision_score

from astropy.coordinates import match_coordinates_sky,SkyCoord, Angle
import astropy.units as u

import warnings
warnings.filterwarnings('ignore')

date = 'June192023'
CC_Webb_Classified = pd.read_csv(f'CC_Classified_DAOPHOT_{date}.csv')
dao_IR = pd.read_csv(f'DAOPHOT_Catalog_{date}_IR.csv')

date = 'DAOPHOT_'+ date

filters = [c for c in CC_Webb_Classified.columns if c[0] == "f"]
print(filters)
fcd_columns = [c for c in CC_Webb_Classified.columns if c[0] == "f" or c[0]=='δ' or c[0]=='(' or c=='Sum1' or (('slope' in c) and c[0]!='e')]
print(CC_Webb_Classified.columns)
errs = ["e_"+f for f in fcd_columns]
bands = fcd_columns+errs

# ----------------------------------------------------------------------------
# Update threshold
thresh = 0.9
CC_Webb_Classified.loc[CC_Webb_Classified.Prob_RF<thresh,'Class_RF'] = 1
CC_Webb_Classified.loc[CC_Webb_Classified.Prob_PRF<thresh,'Class_PRF'] = 1
print(len(CC_Webb_Classified[CC_Webb_Classified.Class_RF==0]),len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0]))
# plt.scatter(CC_Webb_Classified.dropna(subset='Init_Class').Prob_PRF,dao_IR.Prob.values)
# plt.xlabel('Prob PRF')
# plt.ylabel('Prob Init')
# plt.show()
# 
# Print classification reports
print("RF Classification Report")
print(classification_report(CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns).Init_Class,CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns).Class_RF))

print("PRF Classification Report")
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
        y_or_s = f'YSO {m} - PRF and RF'
    elif df_tmp[['Class_RF']].values[0]==0:
        y_or_s = f'YSO {m} - RF'
    elif df_tmp[['Class_PRF']].values[0]==0:
        y_or_s = f'YSO {m} - PRF'
    else:
        y_or_s = f'C - {m}'
    if m in r_inds:
        if m in sp_inds:
            tab_preds.write(f"{df_tmp.Survey} - SPICY {df_tmp.SPICY_ID} & {reit_name[i]} & {y_or_s} \\\ \n")
        else:
            tab_preds.write(f"- & {reit_name[i]} & {y_or_s} \\\ \n")
    else:
        tab_preds.write(f"{df_tmp.Survey} - SPICY {df_tmp.SPICY_ID} & - & {y_or_s} \\\ \n")

tab_preds.close()

print("Table of comparison to other works completed!")

# #TMP - Make matches to Reiter have those RA/DEC
# CC_Webb_Classified.iloc[r_inds].RA = r_1.ra
# CC_Webb_Classified.iloc[r_inds].DEC = r_1.dec

# #----------------------------------------------------------------------------
# Latex table of YSOs for paper
columns = ['RA', 'DEC'] + filters + ['Prob_PRF','Prob_RF']
new_cols = ['RA', 'DEC'] + [f.upper() for f in filters] + ['Prob YSO PRF', 'Prob YSO RF'] #'F770w', 'F1130w', 'F1280w', 'F1800w',

csv_yso = CC_Webb_Classified[(CC_Webb_Classified.Class_RF==0)|(CC_Webb_Classified.Class_PRF==0)].reset_index()
csv_yso = csv_yso.loc[csv_yso.isnull().sum(1).sort_values(ascending=1).index,columns]


file_out = 'CC_Classified_DAOPHOT_latex_ysos.txt'
print("Printing to ", file_out)
f = open(file_out,'w')

for j in range(0,len(columns)):
    f.write(new_cols[j]+'&')
f.write('\\hline \\\ \n')
for i in range(0,len(csv_yso)):
    for j in range(0,len(columns)):
        str_tmp = str("%.4f" %csv_yso[columns[j]].values[i])+"&"
        f.write(str_tmp)
    f.write('\\\ \n')

f.close()
# #----------------------------------------------------------------------------

# Make Plots, define plot details
plt.rcParams['figure.dpi'] = 300
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

prf_col = 'maroon'
rf_col = 'salmon'
colormap = 'Greys'
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
ax.scatter(num_yso_rf,max_f1_rf,c=rf_col,label='RF')
ax.scatter(num_yso_prf,max_f1_prf,c=prf_col,label='PRF')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ax_histx.hist(num_yso_rf,bins=np.arange(xmin,xmax,5),color=rf_col,histtype='step')#
ax_histx.hist(num_yso_prf,bins=np.arange(xmin,xmax,5),color=prf_col,histtype='step')#
ax_histy.hist(max_f1_rf,bins=np.arange(ymin,ymax,0.005),color=rf_col, orientation='horizontal',histtype='step')
ax_histy.hist(max_f1_prf,bins=np.arange(ymin,ymax,0.005),color=prf_col, orientation='horizontal',histtype='step')
ax.set_xlabel('Amount of objects classified as YSOs')
ax.set_ylabel('F1-Score of YSOs')
# ax.set_xscale('log')
ax.legend(loc='upper right')
plt.savefig("Figures/F1-Scoresvs_Num_YSOs_"+date+".png",dpi=300)
plt.close()
print("Plot of f1 scores vs number of YSOs created!")
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

print("Confusion matrices created!")


# #----------------------------------------------------------------------------
# AUC Curve
from sklearn.metrics import RocCurveDisplay
prf_roc = RocCurveDisplay.from_predictions(CC_Webb_Classified.dropna(subset='Init_Class')['Init_Class'],CC_Webb_Classified.dropna(subset='Init_Class')['Prob_PRF'],pos_label=0,name='PRF')
plt.close()
ax = plt.gca()
rf_roc = RocCurveDisplay.from_predictions(CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)['Init_Class'],CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)['Prob_RF'],pos_label=0,name='RF')
plt.close()
rf_roc.plot(ax=ax,c=rf_col)
prf_roc.plot(ax=ax,c=prf_col,alpha=0.8)
# plt.show()
plt.savefig('Figures/ROC_Curve_'+date+'.png',dpi=300)
plt.close()

print('ROC curves created!')
#----------------------------------------
# JWST field image
from astropy.io import fits
from astropy.wcs import WCS

import matplotlib.lines as mlines

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
ra_yso_both = CC_Webb_Classified.RA.values[(CC_Webb_Classified.Class_PRF == 0)&(CC_Webb_Classified.Class_RF == 0)]
dec_yso_both = CC_Webb_Classified.DEC.values[(CC_Webb_Classified.Class_PRF == 0)&(CC_Webb_Classified.Class_RF == 0)]

plt.plot(ra_yso_rf,dec_yso_rf, marker='*',linestyle='none', markersize=15,alpha=0.8,c=rf_col,transform=ax.get_transform('fk5'),label='Our YSOs (RF)')
plt.plot(ra_yso_prf,dec_yso_prf, marker='*', linestyle='none', markersize=15,alpha=0.8,c=prf_col,transform=ax.get_transform('fk5'),label='Our YSOs (PRF)')
plt.plot(ra_yso_both,dec_yso_both, marker='*',linestyle='none', markersize=15,alpha=0.8,fillstyle='left',c=rf_col,markerfacecoloralt=prf_col,markeredgecolor='none',transform=ax.get_transform('fk5'),label='Our YSOs (PRF)')
plt.plot(ra_ir,dec_ir, marker='s',linestyle='none', markersize=15, markeredgecolor='k',fillstyle='none',alpha=0.8,transform=ax.get_transform('fk5'),label='SPICY (2021) or Ohlendorf (2013) YSOs')
plt.plot(ra_1,dec_1, marker='o',linestyle='none', markersize=15,markeredgecolor='k',fillstyle='none', alpha=0.8,transform=ax.get_transform('fk5'),label='Reiter et al. 2022 YSOs')
ax.set_ylim(ymax, ymin)
ax.set_xlim(xmax, xmin)
plt.legend(loc='lower left')
ax.grid(False)
plt.xticks()
plt.yticks()
plt.xlabel('RA')
plt.ylabel('DEC')
plt.tight_layout()
plt.savefig(f"Figures/field_image_{filter}_"+date+".png",dpi=300)
plt.close()

print("Field image created!")
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


axs[0][1].hist([np.count_nonzero(~CC_Webb_Classified[filters].iloc[i].isna()) for i in a],bins=np.arange(1,11,1))
axs[1][1].hist([np.count_nonzero(~CC_Webb_Classified[filters].iloc[i].isna()) for i in b],bins=np.arange(1,11,1))
axs[2][1].hist([np.count_nonzero(~CC_Webb_Classified[filters].iloc[i].isna()) for i in c],bins=np.arange(1,11,1))
axs[3][1].hist([np.count_nonzero(~CC_Webb_Classified[filters].iloc[i].isna()) for i in d],bins=np.arange(1,11,1))
axs[2][1].set_xlim(0,11)
axs[3][1].set_xlabel('Bands Available')

plt.savefig('Figures/seds_'+date+'.png',dpi=300)
plt.close()
print("Plot of SEDs created!")
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

cuts = np.arange(0.0,1.0,0.01)
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

fig, ax = plt.subplots(2,1,dpi=300,figsize=(5,10))

ax[0].plot(cuts,f1s_rf,c=rf_col,label='F1-Score (RF)')
ax[0].plot(cuts,pres_rf,'--',c=rf_col,label='Precision (RF)')
ax[0].plot(cuts,recs_rf,'-.',c=rf_col,label='Recall (RF)')

ax[0].plot(cuts,f1s_prf,c=prf_col,label='F1-Score (PRF)')
ax[0].plot(cuts,pres_prf,'--',c=prf_col,label='Precision (PRF)')
ax[0].plot(cuts,recs_prf,'-.',c=prf_col,label='Recall (PRF)')
# ax[0].set_xlabel('Probability YSO Cut')
ax[1].set_xlabel('Probability YSO Cut')
ax[0].set_ylabel('Metric Score')
ax[1].set_xticks(np.arange(0,1.1,0.1))
ax[0].set_xticks(np.arange(0,1.1,0.1))
ax[0].set_yticks(np.arange(0,1.1,0.1))

# ax2 = ax.twinx()
ax[1].plot(cuts[:-1],nums_rf[:-1],c=rf_col,label='Number YSOs (RF)')
ax[1].plot(cuts[:-1],nums_prf[:-1],linestyle='--',c=prf_col,label='Number YSOs (PRF)')
ax[1].set_ylabel('Number of YSOs')
# ax[1].grid(False)  
ax[1].set_yscale('log')

# lns = [a1, a2, a3, a4]
ax[0].legend(loc='lower left')
ax[1].legend(loc='lower left')
plt.savefig('Figures/Prob_YSO_vs_metric_'+date+'.png',dpi=300)
plt.close()


print("Plot of number of YSOs and all metrics created!")
# --------------------------------------------------------------------
# Histogram of brightnesses with wavelength

fig, axs = plt.subplots(3,2,figsize=(6,8),dpi=300)
fig.tight_layout()
bins =np.arange(6,24,2)
f = 0
for i in range(0,3):
    for j in range(0,2):
        axs[i][j].hist(CC_Webb_Classified.loc[CC_Webb_Classified.Class_RF==0,filters[f]],bins=bins,label=filters[f])
        axs[i][j].set_title(filters[f])
        f += 1
plt.savefig("Figures/Brightness_Band_YSO_"+date+".png",dpi=300)
plt.close()

print("Histogram of brightnesses with wavelength created!")
#  --------------------------------------------------------------------
# CMDs and CCDs with labelled YSOs


for f in fcd_columns:
    fig = plt.subplots(dpi=300)
    err = np.sqrt(CC_Webb_Classified['e_f090w-f444w'].values**2+CC_Webb_Classified['e_'+f].values**2)
    CC_tmp = CC_Webb_Classified[err<0.2]

    plt.scatter(CC_tmp['f090w-f444w'], CC_tmp[f],c=err[err<0.2],marker='.', s=1,cmap=colormap+'_r')
    plt.colorbar(label='Propogated Error')

    CC_rf = CC_tmp.copy()
    CC_rf = CC_rf[CC_rf.Class_RF==0]
    plt.scatter(CC_rf['f090w-f444w'],CC_rf[f],marker='s',s=15,c=rf_col,label='RF YSOs')
    CC_prf = CC_tmp.copy()
    CC_prf = CC_prf[CC_prf.Class_PRF==0]
    plt.scatter(CC_prf['f090w-f444w'],CC_prf[f],marker='*',s=15,c=prf_col, label = 'PRF YSOs')

    plt.xlabel("F090W-F444W")
    plt.ylabel(f.upper())
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f"Figures/CMD_{f}.png")
    plt.close()


print("CMDs and CCDs with labelled YSOs created!")
# ------------------
# Spitzer-DAOPHOT comparison


from mpl_toolkits.axes_grid1 import make_axes_locatable
cc_match = dao_IR
cc_match.where(cc_match!=99.999,np.nan,inplace=True)

fig, ax = plt.subplots(2,1,dpi=300)
lims = np.arange(8.5,16)
print(len(cc_match[['mag3_6']].values))
err = (cc_match[['d3_6m']].values**2+cc_match[['e_f335m']].values**2)**0.5
c = ax[0].scatter(cc_match[['mag3_6']].values,cc_match[['f335m']].values,s=5,c=err,cmap=colormap)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[0].plot(lims,lims,'k',lw=0.3)
ax[0].set_xlabel('IRAC1')
ax[0].set_ylabel('F335M')
ax[0].invert_xaxis()
ax[0].invert_yaxis()
ax[0].set_yticks(np.arange(8,22,2))
plt.colorbar(mappable=c,cax=cax)

err = (cc_match[['d4_5m']].values**2+cc_match[['e_f444w']].values**2)**0.5
c2 = ax[1].scatter(cc_match[['mag4_5']].values,cc_match[['f444w']].values,s=5,c=err,cmap=colormap)
ax[1].plot(lims,lims,'k',lw=0.3)
ax[1].set_ylim(9,16.5)
ax[1].set_xlabel('IRAC2')
ax[1].set_ylabel('F444W')
ax[1].invert_xaxis()
ax[1].invert_yaxis()
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(mappable=c2,cax=cax)
fig.tight_layout()

plt.savefig("Figures/spitz_dao_comp.png")
plt.close()

print("Spitzer-DAOPHOT comparison created!")

# ------------------------------------------------------------------------
# Surface Density Figure

grid = 0 
# xy = wcs_main.all_world2pix(np.transpose([CC_Webb_Classified.RA,CC_Webb_Classified.DEC]),1)
# x = xy[:,0]
# y = xy[:,1]
X = CC_Webb_Classified.x
Y = CC_Webb_Classified.y
xgrid = np.linspace(min(X),max(X),20)
ygrid = np.linspace(min(Y),max(Y),20)
grid_norm, _, _ = np.histogram2d(X, Y, bins=[xgrid, ygrid])


for i in range(1,1001):
    random_ths = np.random.default_rng().random(len(CC_Webb_Classified))
    CC_tmp_th = CC_Webb_Classified[CC_Webb_Classified.Prob_PRF>random_ths]
    x = CC_tmp_th.x
    y = CC_tmp_th.y
    grid_tmp, _, _ = np.histogram2d(x, y, bins=[xgrid, ygrid])
    grid = grid+grid_tmp
grid = grid/i # Take avg
grid = grid.T/grid_norm.T # Normalize, transpose due to inversion of x and y

plt.pcolormesh(xgrid,ygrid,grid,cmap=colormap)
plt.colorbar(label='Normalized Surface Density of cYSOs')
plt.scatter(CC_Webb_Classified.loc[CC_Webb_Classified.Prob_PRF>0.9,'x'],CC_Webb_Classified.loc[CC_Webb_Classified.Prob_PRF>0.9,'y'],s=10,c='maroon',edgecolors='w',linewidth=0.1,marker='*',label='YSOs (Prob > 90%)')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.legend()
plt.savefig("Figures/Surf_dens_norm.png",dpi=300)
plt.close()

print("Surface density plot saved!")