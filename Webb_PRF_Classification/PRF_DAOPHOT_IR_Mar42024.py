import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from astropy.table import Table
from astropy.io.votable import from_table, writeto
from astropy.coordinates import match_coordinates_sky,SkyCoord, Angle
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from scipy.stats import ttest_ind
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,classification_report,f1_score,recall_score,precision_score, RocCurveDisplay, roc_curve,precision_recall_curve,PrecisionRecallDisplay
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

def ext_cor(av,wv):
    return av*(0.3722*(wv**(-2.070))) # For NIR (Chen and Wang 2019 at 3.1)


date = 'CC_Mar42024'
filepath = "./"+date+"/"
CC_Webb_Classified = pd.read_csv(filepath+"Classified_"+date+"_final_class_with_10_percent_saved.csv")
date = date+ '_with_10_percent_saved'
Path('./Figures/'+date).mkdir(parents=True, exist_ok=True)
filters = [c for c in CC_Webb_Classified.columns if (c[0] == "f") and ("-" not in c)and ("_" not in c)]
print(filters)
fcd_columns = [c for c in CC_Webb_Classified.columns if (('-' in c) and c[0]!='e')]
print(fcd_columns)
errs = ["e_"+f for f in fcd_columns]
bands = fcd_columns+errs

# ----------------------------------------------------------------------------
# Add in column density data per point

col_dens = fits.open("Gum31_new.fits")
CC_Webb_Classified['N(H2)'] = col_dens[0].data.byteswap().newbyteorder()[round(CC_Webb_Classified.y).values.astype(int),round(CC_Webb_Classified.x).values.astype(int)]
CC_Webb_Classified['Av'] = CC_Webb_Classified['N(H2)'].values/1e21

print("Total objects in catalogue: ", len(CC_Webb_Classified))
print("Total YSOs in catalogue: ", len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0]))
# Print classification reports
print("RF Classification Report")
print(classification_report(CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns).Init_Class,CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns).Class_RF))

print("PRF Classification Report")
print(classification_report(CC_Webb_Classified.dropna(subset='Init_Class').Init_Class,CC_Webb_Classified.dropna(subset='Init_Class').Class_PRF))

totobjs = len(CC_Webb_Classified)
totysos = len(CC_Webb_Classified[CC_Webb_Classified.Prob_PRF>0.5])
print(totysos,' total ysos with prob greater than 50%, no extra condition added, in catalog')
finaltotconts = len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==1])
print(finaltotconts, " contaminants in set (condition included)")
totysosfrac = len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0])
totnewysoscompspitz = totysosfrac - len(CC_Webb_Classified[CC_Webb_Classified.Init_Class==0])
totprfysos = totysosfrac-len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0].dropna(subset=filters))
totprfrfysos = len(CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified.Class_RF==0)])
print(totprfrfysos)
totprfrfysos = len(CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified.Prob_RF>0.5)])
totprfnotrfysos = len(CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)& (CC_Webb_Classified.Class_RF==1)])
print(totprfnotrfysos)
totprfnotrfysos = len(CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified.Prob_RF<=0.5)& (CC_Webb_Classified.Class_RF>-1)])

print(f"From our photometry of the JWST Cosmic Cliffs field, we retrieve a total of {totobjs} objects. Of these, {totysosfrac} are cYSOs, i.e., they have a probability greater than 50% of being so based on the PRF model and occur in at least 4 our of 6 aadditional runs. This total includes {totnewysoscompspitz} cYSOs not previously identified from Spitzer data, of which {totprfysos} were identified only by the PRF as they contained missing data and hence could not be classified by the RF model. {totprfrfysos} of the remaining cYSOs are found to be cYSOs with the regular RF with a probability $>50%$, and {totprfnotrfysos} are not found to be cYSOs with the RF to this threshold. In our catalog, we mark the latter {totprfnotrfysos} as being insecurely classified.")


writeto(from_table(Table.from_pandas(CC_Webb_Classified,units={'RA':u.deg,'DEC':u.deg})), f"CC_Mar42024/CC_Classified_"+date+".xml")
mk_tables= False
if mk_tables:
    # ----------------------------------------------------------------------------
    # Make table of Reiter, SPICY, and our own classifications
    # reit_df = pd.read_csv("Reiter2022_cYSOs.csv")


    # tab_preds = open("Table_Reiter_SPICY_YSO"+date+".txt",'w')
    # tab_preds.write("\citet{Kuhn2021} & \citet{Reiter2022} & Our Work \\\ \hline \n")

    # r_inds, sep2d, _ = match_coordinates_sky(SkyCoord(reit_df.RA,reit_df.DEC,unit=u.deg), SkyCoord(CC_Webb_Classified.RA,CC_Webb_Classified.DEC,unit=u.deg), nthneighbor=1, storekdtree='kdtree_sky')
    # sp_inds = CC_Webb_Classified[CC_Webb_Classified.Init_Class==0].index
    # _, inds_of_match = np.unique(np.r_[r_inds,sp_inds], return_index=True)
    # matched_inds = np.r_[r_inds,sp_inds][np.sort(inds_of_match)]

    # for i, m in enumerate(matched_inds):
    #     df_tmp = CC_Webb_Classified.iloc[m]
    #     # df_tmp_ir = dao_IR.iloc[i]
    #     if df_tmp[['Class_RF']].values[0]==0 and df_tmp[['Class_PRF']].values[0]==0:
    #         y_or_s = f'YSO {m} - PRF {"%.2f" %df_tmp.Prob_PRF} and RF {"%.2f" %df_tmp.Prob_RF}'
    #     elif df_tmp[['Class_RF']].values[0]==0:
    #         y_or_s = f'YSO {m} - RF {"%.2f" %df_tmp.Prob_RF}'
    #     elif df_tmp[['Class_PRF']].values[0]==0:
    #         y_or_s = f'YSO {m} - PRF {"%.2f" %df_tmp.Prob_PRF}'
    #     else:
    #         y_or_s = f'C {m} - PRF {"%.2f" %df_tmp.Prob_PRF}'
    #     if m in r_inds:
    #         if m in sp_inds:
    #             tab_preds.write(f"{df_tmp.Survey} - SPICY {df_tmp.SPICY_ID} & {reit_df.loc[i,'Name']} & {y_or_s} \\\ \n")
    #         else:
    #             tab_preds.write(f"- & {reit_df.loc[i,'Name']} & {y_or_s} \\\ \n")
    #     else:
    #         tab_preds.write(f"{df_tmp.Survey} - SPICY {df_tmp.SPICY_ID} & - & {y_or_s} \\\ \n")

    # tab_preds.close()

    # print("Table of comparison to other works completed!")

    # #----------------------------------------------------------------------------
    # Latex table of YSOs for paper
    columns = ['RA', 'DEC'] + filters + ['Prob_PRF','frac_runs_yso']
    new_cols = ['R.A.', 'decl.'] + [f.upper() for f in filters] + ['Prob PRF', 'Frac. Runs'] #'F770w', 'F1130w', 'F1280w', 'F1800w',

    csv_yso = CC_Webb_Classified[(CC_Webb_Classified.Class_RF==0)|(CC_Webb_Classified.Class_PRF==0)].reset_index()
    # csv_yso = csv_yso.loc[csv_yso.isnull().sum(1).sort_values(by='frac_runs_yso',ascending=1).index,columns]
    csv_yso = csv_yso.loc[csv_yso.sort_values('Prob_PRF',ascending=0).index,columns]
    # print([c for c in CC_Webb_Classified.columns if ('frac' in c)])

    file_out = filepath+'Classified_latex_ysos_'+date+'.txt'
    print("Printing to ", file_out)
    f = open(file_out,'w')
    f.write("\\begin{deluxetable*}{cccccccccc} \n")
    f.write("\\tablecaption{A sample of the fluxes, J2000 co-ordinates, and probabilities for sources classified as cYSOs. \\label{tab:big_tab}} \n")
    f.write("\\tablehead{")
    for j in range(0,len(columns)):
        f.write("\colhead{"+new_cols[j]+'}&')
    f.write('} \n')
    f.write('\\startdata \n')
    for i in range(0,len(csv_yso)):
        for j in range(0,len(columns)):
            if j < 2:
                str_tmp = str("%.4f" %csv_yso[columns[j]].values[i])+"&"
            elif j == (len(columns)-1):
                str_tmp = str("%.2f" %csv_yso[columns[j]].values[i])
            else:
                str_tmp = str("%.2f" %csv_yso[columns[j]].values[i])+"&"
            f.write(str_tmp)
        f.write('\\\ \n')
    f.write("\\enddata \n")
    f.write("\\tablecomments{A probability less than 0.50 meant the object was not classified as a YSO in that algorithm, and a probability of -99.999 meant the object had missing data and so was unable to be classified by the RF model. \\textbf{Full catalog available at...} \n}")
    f.write("\n \\end{deluxetable*} \n")
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

remake_figs = input("Remake metric figures? (y) or (n) ")
rerun_contours = input("Re-run SFR/MGAS/NPS within contour calculations? (y) or (n) ")
rerun_sd = input('Re-run Surface density computation? (y) or (n)')
rerun_sd_subs = input('Re-run Surface density computation for sub-stellar objects? (y) or (n)')
calc_prf_sd = input("Calculate SFR for PRF results? (y) or (n) ")


if remake_figs == 'y':
    print('Making f1-scores...')
    # #----------------------------------------------------------------------------
    # Scatter plot with hists for number of YSOs vs F1-Score
    num_yso_rf = np.loadtxt(f"{filepath}Num_YSOs_RF_{date}")
    num_yso_prf = np.loadtxt(f"{filepath}Num_YSOs_PRF_{date}")
    max_f1_rf = np.loadtxt(f"{filepath}Max_f1s_RF_{date}")
    max_f1_prf = np.loadtxt(f"{filepath}Max_f1s_PRF_{date}")

    # print('Mean number of YSOs:',np.mean(num_yso), 'Median number of YSOs:', np.median(num_yso))
    # print('Mean F1-Score:',np.mean(max_f1), 'Median F1-Score:', np.median(max_f1), 'Standard deviation F1-Score:', np.std(max_f1))
    fig = plt.figure(figsize=(8, 6),dpi=300)
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

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.scatter(num_yso_rf,max_f1_rf,c=rf_col,label='RF')
    ax.scatter(num_yso_prf,max_f1_prf,c=prf_col,label='PRF')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = 0.95, 1.0
    ax_histx.hist(num_yso_rf,bins=np.arange(xmin,xmax,5),color=rf_col,histtype='step')#
    ax_histx.hist(num_yso_prf,bins=np.arange(xmin,xmax,5),color=prf_col,histtype='step')#
    ax_histy.hist(max_f1_rf,bins=np.arange(ymin,ymax,0.005),color=rf_col, orientation='horizontal',histtype='step')
    ax_histy.hist(max_f1_prf,bins=np.arange(ymin,ymax,0.005),color=prf_col, orientation='horizontal',histtype='step')
    ax.set_xlabel('Amount of objects classified as YSOs')
    ax.set_ylabel('F1-Score of YSOs')
    # ax.set_xscale('log')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"Figures/{date}/F1-Scores_vs_Num_YSOs_"+date+".png",dpi=300)
    plt.close()
    print("Plot of f1 scores vs number of YSOs created!")

    # #----------------------------------------
    # # confusion matrix


    tar_va = CC_Webb_Classified.dropna(subset='Init_Class').dropna(subset=fcd_columns)[['Init_Class']].values.astype(int)
    pred_va = CC_Webb_Classified.dropna(subset='Init_Class').dropna(subset=fcd_columns)[['Class_RF']].values
    ConfusionMatrixDisplay.from_predictions(tar_va,pred_va,cmap='Reds',display_labels=['YSO', 'Contaminant'])
    plt.grid(False)
    plt.tight_layout()
    plt.yticks(rotation=45)
    plt.savefig(f'Figures/{date}/CM_va_RF_SPICY_'+date+'.png',dpi=300,facecolor=fig.get_facecolor())
    plt.close()

    tar_va = CC_Webb_Classified.dropna(subset='Init_Class')[['Init_Class']].values.astype(int)
    pred_va = CC_Webb_Classified.dropna(subset='Init_Class')[['Class_PRF']].values
    ConfusionMatrixDisplay.from_predictions(tar_va,pred_va,cmap='Reds',display_labels=['YSO', 'Contaminant'])
    plt.grid(False)
    plt.tight_layout()
    plt.yticks(rotation=45)
    plt.savefig(f'Figures/{date}/CM_va_PRF_SPICY_'+date+'.png',dpi=300,facecolor=fig.get_facecolor())
    plt.close()

    print("Confusion matrices created!")


    # #----------------------------------------------------------------------------
    # AUC Curve
    roc_class = abs(CC_Webb_Classified.dropna(subset='Init_Class')['Init_Class'].copy().values -1)
    roc_class_rf = abs(CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)['Init_Class'].values -1)
    print("Number of contaminants:",len(roc_class[roc_class==0])," number of YSOs: ",len(roc_class[roc_class==1]))
    prf_roc = RocCurveDisplay.from_predictions(roc_class,CC_Webb_Classified.dropna(subset='Init_Class')['Prob_PRF'],pos_label=1,name='PRF')
    plt.close()
    ax = plt.gca()
    rf_roc = RocCurveDisplay.from_predictions(roc_class_rf,CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)['Prob_RF'],pos_label=1,name='RF')
    plt.close()
    rf_roc.plot(ax=ax,c=rf_col)
    prf_roc.plot(ax=ax,c=prf_col,alpha=0.8)
    ax.set_xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.show()
    plt.savefig(f'Figures/{date}/ROC_Curve_'+date+'.png',dpi=300)
    plt.close()

    fpr, tpr, thresholds = roc_curve(roc_class,CC_Webb_Classified.dropna(subset='Init_Class')['Prob_PRF'],pos_label=1)
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    print('ROC curves created!')

    # -------------------------------------------
    # PR- Curve

    precision, recall, thresholds = precision_recall_curve(roc_class,CC_Webb_Classified.dropna(subset='Init_Class')['Prob_PRF'],pos_label=1)
    prf_pr = PrecisionRecallDisplay.from_predictions(roc_class,CC_Webb_Classified.dropna(subset='Init_Class')['Prob_PRF'],pos_label=1,name='PRF')
    plt.close()
    ax = plt.gca()
    rf_pr = PrecisionRecallDisplay.from_predictions(roc_class_rf,CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns)['Prob_PRF'],pos_label=1,name='RF')
    plt.close()
    rf_pr.plot(ax=ax,c=rf_col)
    prf_pr.plot(ax=ax,c=prf_col,alpha=0.8)
    ax.set_xscale('log')
    plt.savefig(f'Figures/{date}/PR-curve_'+date+".png",dpi=300)
    plt.close()

    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

    #----------------------------------------
    # JWST field image
    # Plot image
    f = fits.open(f'/users/breannacrompvoets/DAOPHOT/NGC3324/f090w_ADU.fits')
    fig, ax = plt.subplots(figsize=(14,8),dpi=300)
    ax = plt.subplot(projection=WCS(f[0].header,relax=False))
    ax.imshow(f[0].data,cmap='gray_r',vmax=1300,origin='lower')

    lon = ax.coords[0]
    lat = ax.coords[1]

    # We need to set the ticklabel positions explicitely because matplolib gets confused with ra and dec (image is rotated 90 degrees wrt coords)
    lon.set_ticklabel_position('l')
    lat.set_ticklabel_position('b')
    lat.set_axislabel('RA (deg)')
    lon.set_axislabel('DEC (deg)')
    lon.set_major_formatter('d.dd')
    lat.set_major_formatter('d.dd')
    lon.set_ticklabel(rotation=90)


    ymax, ymin = ax.get_ylim()
    xmax, xmin = ax.get_xlim()


    matches_csv = pd.read_csv("matches_to_"+'July242023'+".csv")#date
    x_inds = [i for i in matches_csv.index if 'X-RAY' in matches_csv.loc[i,'Name']] 
    sp_inds = [i for i in matches_csv.index if 'SPICY' in matches_csv.loc[i,'Name']] 
    o_inds = [i for i in matches_csv.index if 'OHL' in matches_csv.loc[i,'Name']] 
    r_inds = [i for i in matches_csv.index if ('MHO' in matches_csv.loc[i,'Name']) or ('HH' in matches_csv.loc[i,'Name'])] 

    ra_yso_prf = CC_Webb_Classified.RA.values[CC_Webb_Classified.Class_PRF == 0]
    dec_yso_prf = CC_Webb_Classified.DEC.values[CC_Webb_Classified.Class_PRF == 0]

    plt.scatter(ra_yso_prf,dec_yso_prf,c=CC_Webb_Classified.Prob_PRF.values[CC_Webb_Classified.Class_PRF == 0],cmap='Reds', marker='*', s=100,alpha=0.8,transform=ax.get_transform('fk5'),label='Our YSOs (PRF)')
    plt.colorbar(label='PRF Probability YSO')
    prev_work = ['Preibisch et al 2014 X-Ray', 'Ohlendorf et al. 2013 IR', 'Kuhn et al. 2021 IR', 'Reiter et al. 2022 Outflow Prog.']
    m_prev = ['X','o','s','h']
    for m_i, prev_inds in enumerate([x_inds,o_inds,sp_inds,r_inds]):
        plt.scatter(matches_csv.loc[prev_inds,'RA'],matches_csv.loc[prev_inds,'DEC'],marker=m_prev[m_i],edgecolors='k',facecolors='none',transform=ax.get_transform('fk5'), s=80,label=prev_work[m_i])

    ax.set_ylim(ymax, ymin)
    ax.set_xlim(xmax, xmin)
    plt.legend(loc='lower left')
    ax.grid(False)
    
    scalebar = AnchoredSizeBar(ax.transData,
                            1405, '0.5 pc', 'lower right', 
                            pad=0.5,
                            color='k',
                            frameon=False)

    ax.add_artist(scalebar)

    plt.tight_layout()
    plt.savefig(f"Figures/{date}/field_image_f090w_"+date+".png",dpi=300)
    plt.close()

    print("Field image created!")
    #--------------------------------------------------------------------
    # Compare SEDs and number of bands available for validation set. 
    a = CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified['Init_Class']==0)].index # Correctly classified as YSO
    b = CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified['Init_Class']==1)].index # Incorrectly classified as YSO
    c = CC_Webb_Classified[(CC_Webb_Classified.Class_PRF!=0)&(CC_Webb_Classified['Init_Class']==0)].index # Incorrectly classified as Star
    d = CC_Webb_Classified[(CC_Webb_Classified.Class_PRF!=0)&(CC_Webb_Classified['Init_Class']==1)].index # Correctly classified as Star
    

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

        ax.plot(np.array([webb_mic]*len(cat.iloc[ind])).transpose(),(cat.iloc[ind][webb_bands].to_numpy()+correction).transpose(),'--',c='r',alpha=0.7)
        ax.plot(webb_mic,mu[webb_bands].to_numpy()[0]+correction,**kwargs,c='r',label='Webb SED')
        ax.fill_between(webb_mic,mu[webb_bands].to_numpy()[0]+correction-sig[webb_bands].to_numpy()[0],mu[webb_bands].to_numpy()[0]+correction+sig[webb_bands].to_numpy()[0],color='r',alpha=0.1)
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

    plt.savefig(f'Figures/{date}/seds_'+date+'.png',dpi=300)
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

    # f1s_tot = []
    # recs_tot = []
    # pres_tot = []
    # nums_tot = []

    cuts = np.arange(0.0,1.0,0.01)
    for i in cuts:
        preds = np.array([1]*len(CC_Webb_Classified.dropna(subset=['Init_Class']+fcd_columns).values))
        # print(np.where(prob_yso_rf_ir>i)[0])
        preds[np.where(prob_yso_rf_ir>i)[0]] = 0
        f1s_rf.append(f1_score(tar_va_rf,preds,average=None)[0])
        recs_rf.append(recall_score(tar_va_rf,preds,average=None)[0])
        pres_rf.append(precision_score(tar_va_rf,preds,average=None)[0])
        preds = np.array([1]*len(CC_Webb_Classified['Class_RF'].values))
        preds[np.where(prob_yso_rf>i)[0]] = 0
        nums_rf.append(len(preds[preds==0]))

        preds = np.array([1]*len(CC_Webb_Classified.dropna(subset='Init_Class').values))
        preds[np.where(prob_yso_prf_ir>i)[0]] = 0
        f1s_prf.append(f1_score(tar_va_prf,preds,average=None)[0])
        recs_prf.append(recall_score(tar_va_prf,preds,average=None)[0])
        pres_prf.append(precision_score(tar_va_prf,preds,average=None)[0])
        preds = np.array([1]*len(CC_Webb_Classified))
        preds[np.where(prob_yso_prf>i)[0]] = 0
        nums_prf.append(len(preds[preds==0]))

    fig, ax = plt.subplots(dpi=300,figsize=(5,5))

    ax.plot(cuts,f1s_rf,c=rf_col,label='F1-Score (RF)')
    ax.plot(cuts,pres_rf,'--',c=rf_col,label='Precision (RF)')
    ax.plot(cuts,recs_rf,'-.',c=rf_col,label='Recall (RF)')

    ax.plot(cuts,f1s_prf,c=prf_col,label='F1-Score (PRF)')
    ax.plot(cuts,pres_prf,'--',c=prf_col,label='Precision (PRF)')
    ax.plot(cuts,recs_prf,'-.',c=prf_col,label='Recall (PRF)')

    # ax.plot(cuts,f1s_prf,c='k',label='F1-Score (Final)')
    # ax.plot(cuts,pres_prf,'--',c='k',label='Precision (Final)')
    # ax.plot(cuts,recs_prf,'-.',c='k',label='Recall (Final)')
    
    ax.vlines(x=0.5, ymin = 0, ymax = max(recs_prf),colors=['grey'],linestyle='dotted',alpha=0.9)
    ax.set_xlabel('Probability YSO Cut')
    ax.set_ylabel('Metric Score')
    ax.set_xticks(np.arange(0,1.1,0.1))
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.legend(loc='lower left')
    plt.savefig(f'Figures/{date}/Prob_YSO_vs_metric_'+date+'.png',dpi=300)
    plt.close()



    # ax2 = ax.twinx()
    fig, ax = plt.subplots(dpi=300,figsize=(5,5))
    ax.hlines(y=630, xmin = 0, xmax = 1.0,colors=['grey'],linestyle='dotted',alpha=0.5)
    ax.vlines(x=0.5, ymin = 0, ymax = max(nums_prf),colors=['grey'],linestyle='dotted',alpha=0.9)
    # ax.vlines(x=0.67, ymin = 0, ymax = max(nums_prf), colors=['grey'],linestyle='dotted',alpha=0.5)
    # ax.vlines(x=0.9, ymin = 0, ymax = max(nums_prf), colors=['grey'],linestyle='dotted',alpha=0.5)
    ax.plot(cuts[:-1],nums_rf[:-1],c=rf_col,label='Number YSOs (RF)')
    ax.plot(cuts[:-1],nums_prf[:-1],linestyle='--',c=prf_col,label='Number YSOs (PRF)')
    # ax.plot(cuts[:-1],nums_tot[:-1],linestyle=':',c='k',label='Number YSOs (PRF)')
    ax.set_ylabel('Number of YSOs')
    ax.set_xlabel('Probability YSO Cut')
    ax.set_xticks(np.arange(0,1.1,0.1))
    # ax[1].grid(False)  
    ax.set_yscale('log')

    # lns = [a1, a2, a3, a4]
    ax.legend(loc='lower left')
    plt.savefig(f'Figures/{date}/Prob_YSO_vs_number_'+date+'.png',dpi=300)
    plt.close()



    fig, ax = plt.subplots(dpi=300,figsize=(5,5))
    # ax.plot(cuts[:-1],nums_rf[:-1],c=rf_col,label='Number YSOs (RF)')
    rel_perf = (np.array(nums_prf[:-1])/np.array(nums_rf[:-1]))*(np.array(pres_prf[:-1])/np.array(pres_rf[:-1]))
    # rel_perf_tot = (np.array(nums_tot[:-1])/np.array(nums_rf[:-1]))*(np.array(pres_tot[:-1])/np.array(pres_rf[:-1]))
    ax.vlines(x=0.5, ymin = 0, ymax = max(rel_perf),colors=['grey'],linestyle='dotted',alpha=0.9)
    ax.plot(cuts[:-1],rel_perf,linestyle='--',c=prf_col,label='PRF-RF Relative Performance')
    # ax.plot(cuts[:-1],rel_perf_tot,linestyle='--',c=prf_col,label='Final-RF Relative Performance')
    ax.set_ylabel('Relative Performance')
    ax.set_xlabel('Probability YSO Cut')
    ax.set_xticks(np.arange(0,1.1,0.1))
    # ax[1].grid(False)  
    # ax.set_yscale('log')
    print((np.array(nums_prf[:-1])/np.array(nums_rf[:-1])*(np.array(pres_prf[:-1])/np.array(pres_rf[:-1])))[49])

    # lns = [a1, a2, a3, a4]
    ax.legend(loc='lower left')
    plt.savefig(f'Figures/{date}/Prob_YSO_vs_rel_perf_'+date+'.png',dpi=300)
    plt.close()


    print("Plot of number of YSOs and all metrics created!")

#--------------------------------------------------------------------------------------------------------
# Isochrones plot
parsec = pd.read_csv('~/Downloads/output622966404146.dat.txt',comment='#',delim_whitespace=True) # tot av = 0, jwst_mags (not nv 22), parsec 2.0, ωi = 0.0
p_tmp = parsec.copy()[(parsec.logAge==6.32222)&(parsec.label==0)]

filt = 'f200w'
xaxis_col = 'f200w-f335m'
# xaxis_col = 'f200w-f444w'

err = np.sqrt(CC_Webb_Classified['e_'+xaxis_col].values**2+CC_Webb_Classified['e_'+filt].values**2)

plt.scatter(CC_Webb_Classified.loc[err<0.2,xaxis_col], CC_Webb_Classified.loc[err<0.2,filt],c=err[err<0.2],marker='.', s=1,cmap=colormap+'_r')
plt.colorbar(label='Propogated Error')
# plt.scatter(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0].f090w-CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0].f200w, CC_Webb_Classified.loc[CC_Webb_Classified.Class_PRF==0,filt],s=15,marker='*',c=prf_col,label='cYSOs')

cols = ['lightcoral','indianred','firebrick','maroon']
i=0
m = []

wv_f1 = int(xaxis_col.split('-')[0][1:-1])/100
wv_f2 = int(xaxis_col.split('-')[1][1:-1])/100
for av in np.arange(0.5,14.5,4):
    plt.plot(p_tmp[xaxis_col.split('-')[0].upper()+'_fSBmag']+ext_cor(av,wv_f1)-p_tmp[xaxis_col.split('-')[1].upper()+'_fSBmag']-ext_cor(av,wv_f2),p_tmp[filt.upper()+'_fSBmag']+5*np.log10(2350)-5+ext_cor(av,int(filt[1:-1])/100),label='$A_V$ = '+str(av)+' mag',c=cols[i])
    i+=1
    m.append([np.array(p_tmp[xaxis_col.split('-')[0].upper()+'_fSBmag']+ext_cor(av,wv_f1)-p_tmp[xaxis_col.split('-')[1].upper()+'_fSBmag']-ext_cor(av,wv_f2))[0],np.array(p_tmp[filt.upper()+'_fSBmag']+5*np.log10(2350)-5+ext_cor(av,int(filt[1:-1])/100))[0]])

# Compute substellar objects
def slope_from_av(filt,col,pc):
    wv_f = int(filt[1:-1])/100
    wv_f1 = int(col.split('-')[0][1:-1])/100
    wv_f2 = int(col.split('-')[1][1:-1])/100
    rise = ext_cor(av=0,wv=wv_f)-ext_cor(av=14,wv=wv_f)
    run = ext_cor(av=0,wv=wv_f1)-ext_cor(av=14,wv=wv_f1)-(ext_cor(av=0,wv=wv_f2)-ext_cor(av=14,wv=wv_f2))
    slope = rise/run
    b = slope*(pc[col.split('-')[0].upper()+'_fSBmag'].values[0]-pc[col.split('-')[1].upper()+'_fSBmag'].values[0])-(pc[filt.upper()+'_fSBmag'].values[0]+5*np.log10(2350)-5)
    return slope, b

slope_col = 'f200w-f335m'
slope_f = 'f200w'
slope, b = slope_from_av(filt=slope_f,col=slope_col,pc=p_tmp)

sub_yso_pop_bool = (CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified[slope_f] >= (slope*CC_Webb_Classified[slope_col]-b))#(CC_Webb_Classified['f200w'] >= (slope1*CC_Webb_Classified['f090w-f200w']-b1))|(CC_Webb_Classified['f444w'] >= (slope2*CC_Webb_Classified['f335m-f444w']-b2))|(CC_Webb_Classified['f444w'] >= (slope3*CC_Webb_Classified['f200w-f444w']-b3)))
fp_pop_bool = (CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified['f090w-f200w']<2)
cont_bl_pop_bool = (CC_Webb_Classified.Class_PRF==1)&(CC_Webb_Classified['f090w-f200w']<2)
yso_pop_bool = (CC_Webb_Classified.Class_PRF==0)&~sub_yso_pop_bool&~fp_pop_bool
cont_ll_pop_bool = (CC_Webb_Classified.Class_PRF==1)&(CC_Webb_Classified[slope_f] >= (slope*CC_Webb_Classified[slope_col]-b))#((CC_Webb_Classified['f200w'] >= (slope1*CC_Webb_Classified['f090w-f200w']-b1))|(CC_Webb_Classified['f444w'] >= (slope2*CC_Webb_Classified['f335m-f444w']-b2))|(CC_Webb_Classified['f444w'] >= (slope3*CC_Webb_Classified['f200w-f444w']-b3)))
cont_hl_pop_bool = (CC_Webb_Classified.Class_PRF==1)&~cont_ll_pop_bool
yso_pop_sp_bool = (CC_Webb_Classified.Init_Class==0)
cont_pop_sp_bool = (CC_Webb_Classified.Init_Class==1)
plt.legend()

CC_prf = CC_Webb_Classified.loc[yso_pop_bool].copy()
plt.scatter(CC_prf[xaxis_col],CC_prf[filt],marker='*',s=25,c=prf_col,edgecolors=rf_col,linewidths=0.3, label = 'PRF cYSOs')

CC_subs = CC_Webb_Classified.loc[sub_yso_pop_bool].copy()
plt.scatter(CC_subs[xaxis_col],CC_subs[filt],marker='s',s=5,c=rf_col,edgecolors=prf_col,linewidths=0.3, label = 'Sub-stellar cYSOs')

CC_sp = CC_Webb_Classified.loc[yso_pop_sp_bool].copy()
plt.scatter(CC_sp[xaxis_col],CC_sp[filt],marker='o',s=10,c='navy', label = 'Spitzer YSOs')

CC_fp = CC_Webb_Classified.loc[fp_pop_bool].copy()
plt.scatter(CC_fp[xaxis_col],CC_fp[filt],marker='o',s=5,c='lightskyblue', edgecolors='blue', label = 'Probable False-Positives')

plt.arrow(m[0][0],m[0][1],m[-1][0]-m[0][0],m[-1][1]-m[0][1],width=0.1,color='r',label='M = 0.09 $M_\odot$',fc='w',ec='k',lw=0.35)

plt.xlabel(xaxis_col.upper())
plt.ylabel(filt.upper())
plt.gca().invert_yaxis()
plt.xlim((-1.5,6.2))
plt.xticks([0,2,4,6])
plt.savefig(f'Figures/{date}/isochrones_'+date+'.png',dpi=300)
plt.close()

frac_substel = len(CC_Webb_Classified.loc[sub_yso_pop_bool])/(len(CC_Webb_Classified.loc[sub_yso_pop_bool|yso_pop_bool]))
print("Fraction of objects that are sub-stellar: ", frac_substel)
print("Number of objects that are sub-stellar: ", len(CC_Webb_Classified.loc[sub_yso_pop_bool]))
print("There are ",len(CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)&(CC_Webb_Classified['f200w-f444w'].isna())])," NaN points in sub-stellar check.")
print("There are ",len(CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)&(~CC_Webb_Classified['f200w-f444w'].isna())])," non-NaN points in sub-stellar check.")
print("There are ",len(CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)])," total points in sub-stellar check.")



#  --------------------------------------------------------------------
# CMDs and CCDs with labelled YSOs
plt.rcParams['font.size'] = 5
fig, axs = plt.subplots(3,3,dpi=300)
i=j=0

xaxis_col = 'f090w-f200w'
for f in ['f090w','f187n','f200w','f335m','f444w','f470n','f187n-f444w','f187n-f470n','f200w-f335m']:
    err = np.sqrt(CC_Webb_Classified['e_'+xaxis_col].values**2+CC_Webb_Classified['e_'+f].values**2)
    CC_tmp = CC_Webb_Classified[err<0.2]

    col_scale = axs[i][j].scatter(CC_tmp[xaxis_col], CC_tmp[f],c=err[err<0.2],marker='.', s=0.1,cmap=colormap+'_r')
    # plt.colorbar(label='Propogated Error')

    axs[i][j].set_xlabel(xaxis_col.upper())
    axs[i][j].set_ylabel(f.upper())
    if len(f) < 6:
        axs[i][j].invert_yaxis()
    # plt.savefig(f"Figures/{date}/CMD_{f}.png")


    CC_prf = CC_Webb_Classified.loc[yso_pop_bool&(err<0.2)].copy()
    axs[i][j].scatter(CC_prf[xaxis_col],CC_prf[f],marker='*',s=25,c=prf_col,edgecolors=rf_col,linewidths=0.3, label = 'PRF cYSOs')


    CC_subs = CC_Webb_Classified.loc[sub_yso_pop_bool&(err<0.2)].copy()
    axs[i][j].scatter(CC_subs[xaxis_col],CC_subs[f],marker='s',s=5,c=rf_col,edgecolors=prf_col,linewidths=0.3, label = 'Sub-stellar cYSOs')

    CC_sp = CC_Webb_Classified.loc[yso_pop_sp_bool&(err<0.2)].copy()
    axs[i][j].scatter(CC_sp[xaxis_col],CC_sp[f],marker='o',s=10,c='navy', label = 'Spitzer YSOs')


    CC_fp = CC_Webb_Classified.loc[fp_pop_bool&(err<0.2)].copy()
    # axs[i][j].scatter(CC_fp[xaxis_col],CC_fp[f],marker='o',s=5,c='lightskyblue', edgecolors='blue', label = 'Probable False-Positives')
    
    if f == "f335m-f444w":
        axs[i][j].set_ylim(-1,3)
    if f == "f200w-f335m":
        axs[i][j].set_ylim(-1,5)
    if f == "f444w-f470n":
        axs[i][j].set_ylim(-3,1)
    
    j+=1
    if j == 3:
        j = 0
        i+=1
    if i == 3:
        i = 0

fig.tight_layout()
fig.subplots_adjust(right=0.87)
cbar_ax = fig.add_axes([0.90, 0.05, 0.01, 0.9])
fig.colorbar(col_scale,label="Propogated Error", cax=cbar_ax)
axs[0][2].legend(loc='upper right')
plt.savefig(f"Figures/{date}/CMDs_all_{date}.png")
plt.close()


print("CMDs and CCDs with labelled YSOs created!")


keep_substel = input("Keep sub-stellar component?")
if keep_substel=='y':
    substel = 'corr_substellar_'
    print('Keeping and correcting mass for substellar objects.')
else:
    substel = 'no_substellar_'
    CC_Webb_Classified.loc[sub_yso_pop_bool,'Class_PRF'] = 1
    print('Removing substellar objects.')
# Exclude probable contaminants
print("There are also ", len(CC_Webb_Classified.loc[fp_pop_bool])," blue false positives")
CC_Webb_Classified.loc[CC_Webb_Classified['f090w-f200w']<2,'Class_PRF'] = 1
print("This leaves ",len(CC_Webb_Classified.loc[CC_Webb_Classified.Class_PRF==0])," cYSOs after the spurious population has been removed.")
plt.rcParams['font.size'] = 12
# ------------------
# Spitzer-DAOPHOT comparison


# from mpl_toolkits.axes_grid1 import make_axes_locatable
# cc_match = dao_IR.copy().loc[abs(dao_IR.f444w-dao_IR.mag4_5)<0.5]
# cc_match.where(cc_match!=99.999,np.nan,inplace=True)

# fig, ax = plt.subplots(2,1,dpi=300)
# lims = np.arange(8.5,16)
# print(len(cc_match[['mag3_6']].values))
# err = (cc_match[['d3_6m']].values**2+cc_match[['e_f335m']].values**2)**0.5
# c = ax[0].scatter(cc_match[['mag3_6']].values,cc_match[['f335m']].values,s=5,c=cc_match.Class,cmap=colormap+'_r')
# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# ax[0].plot(lims,lims,'k',lw=0.3)
# ax[0].plot(lims,lims+0.5,'k',lw=0.3)
# ax[0].plot(lims,lims-0.5,'k',lw=0.3)
# ax[0].set_xlabel('IRAC1')
# ax[0].set_ylabel('F335M')
# ax[0].invert_xaxis()
# ax[0].invert_yaxis()
# ax[0].set_yticks(np.arange(8,22,2))
# plt.colorbar(mappable=c,cax=cax)

# err = (cc_match[['d4_5m']].values**2+cc_match[['e_f444w']].values**2)**0.5
# c2 = ax[1].scatter(cc_match[['mag4_5']].values,cc_match[['f444w']].values,s=5,c=cc_match.Class,cmap=colormap+'_r')
# ax[1].plot(lims,lims,'k',lw=0.3)
# ax[1].plot(lims,lims+0.5,'k',lw=0.3)
# ax[1].plot(lims,lims-0.5,'k',lw=0.3)
# ax[1].set_ylim(9,16.5)
# ax[1].set_xlabel('IRAC2')
# ax[1].set_ylabel('F444W')
# ax[1].invert_xaxis()
# ax[1].invert_yaxis()
# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# plt.colorbar(mappable=c2,cax=cax)
# fig.tight_layout()

# plt.savefig(f"Figures/{date}/spitz_dao_comp.png")
# plt.close()

# print("Spitzer-DAOPHOT comparison created!")



# ------------------------------------------------------------------------
# Surface/Column Density Figure

# Column density
col_dens = fits.open("Gum31_new.fits")
cdata = col_dens[0].data
w=80
col_dens_dat = col_dens[0].data[::w,::w]
levels = np.arange(round(min(col_dens_dat.ravel())/1e22,2)*1e22,max(col_dens_dat.ravel()),5e20) # Approximately 5e20 to 1.4e22, steps of 5e20 or 0.5 mag Av
levels[:-10] # Keep only layers until there are no longer anymore YSOs within them
x_col = np.arange(np.shape(cdata)[1])
y_col = np.arange(np.shape(cdata)[0])

# Various conversion factors
pix_to_parsec_x = 178/(2*2350*np.tan(7.4/120*np.pi/180)) # pixels/pc in the x direction (Herscehl res)
pix_to_parsec_y = 106/(2*2350*np.tan(4.4/120*np.pi/180)) # pixels/pc in the y direction (Herscehl res)
pix_to_parsec2 = pix_to_parsec_x*pix_to_parsec_y
pix_to_parsec_x_full = 14125/(2*2350*np.tan(7.4/120*np.pi/180)) # pixels/pc in the x direction (JWST res)
pix_to_parsec_y_full = 8421/(2*2350*np.tan(4.4/120*np.pi/180)) # pixels/pc in the y direction (JWST res)
pix_to_parsec2_full = pix_to_parsec_x_full*pix_to_parsec_y_full
cm2_to_pix = (3.086e18)**2/pix_to_parsec2
nh2_to_m_gas = 2*(1.67e-24/1.98847e33)/0.71 # N(H2) to solar mass
# sd_to_sfr = 0.5/2

grid = 0 
grid_pix = (178,106) #Pixel size corresponding to approximate Herschel resolution.

tic = time.perf_counter()
X = CC_Webb_Classified.x
Y = CC_Webb_Classified.y
xgrid = np.linspace(min(X),max(X),grid_pix[0])
ygrid = np.linspace(min(Y),max(Y),grid_pix[1])

if rerun_sd == 'y':
    n=11
    grid_nnd = np.empty((len(xgrid),len(ygrid)))
    grid_nnd_norm = np.empty((len(xgrid),len(ygrid)))
    grid_nnd_e = np.empty((len(xgrid),len(ygrid)))
    half_xcell = (xgrid[1]-xgrid[0])/2          
    half_ycell = (ygrid[1]-ygrid[0])/2
    CC_yso_tmp = CC_Webb_Classified.loc[(CC_Webb_Classified.Class_PRF==0)]
    for xg in range(len(xgrid)):
        for yg in range(len(ygrid)):
            xx = xgrid[xg]+half_xcell
            yy = ygrid[yg]+half_ycell
            dN_norm = np.sort(np.sqrt((xx-CC_Webb_Classified.x)**2+(yy-CC_Webb_Classified.y)**2),axis=None)[n-1]
            grid_nnd_norm[xg,yg] = ((n-1)/(np.pi*dN_norm**2))*(pix_to_parsec2_full/pix_to_parsec2) 
            
            nnd_tmp = []
            for i in range(1,101):
                random_ths = np.random.default_rng().random(len(CC_yso_tmp))
                CC_tmp_th = CC_yso_tmp[(CC_yso_tmp.Prob_PRF>random_ths)]
                dN = np.sort(np.sqrt((xx-CC_tmp_th.x)**2+(yy-CC_tmp_th.y)**2),axis=None)[n-1]
                nnd_tmp.append((n-1)/(np.pi*dN**2))
            grid_nnd[xg,yg] = np.mean(nnd_tmp)*(pix_to_parsec2_full/pix_to_parsec2)  
            grid_nnd_e[xg,yg] = np.std(nnd_tmp)*(pix_to_parsec2_full/pix_to_parsec2)                      
    
    grid = grid_nnd
    grid_e = grid_nnd_e
    grid_to_norm = grid_nnd_norm

    toc = time.perf_counter()
    print(f"Completed contour searching in {(toc - tic)/60:0.2f} minutes!\n\n")

    np.save("sd_grid_nnd_"+substel+date+"", grid)
    np.save("sd_grid_nnd_e_"+substel+date+"", grid_e)
    np.save("sd_grid_norm_nnd_"+substel+date+"",grid_to_norm)
else: 
    grid = np.load("sd_grid_nnd_"+substel+date+".npy")
    grid_e = np.load("sd_grid_nnd_e_"+substel+date+".npy")
    grid_to_norm = np.load("sd_grid_norm_nnd_"+substel+date+".npy")



if rerun_sd_subs == 'y':
    n=11
    grid_nnd = np.empty((len(xgrid),len(ygrid)))
    grid_nnd_norm = np.empty((len(xgrid),len(ygrid)))
    grid_nnd_e = np.empty((len(xgrid),len(ygrid)))
    half_xcell = (xgrid[1]-xgrid[0])/2          
    half_ycell = (ygrid[1]-ygrid[0])/2
    CC_yso_tmp = CC_Webb_Classified.loc[sub_yso_pop_bool]
    CC_Webb_tmp = CC_Webb_Classified.loc[~yso_pop_bool]
    for xg in range(len(xgrid)):
        for yg in range(len(ygrid)):
            xx = xgrid[xg]+half_xcell
            yy = ygrid[yg]+half_ycell
            dN_norm = np.sort(np.sqrt((xx-CC_Webb_tmp.x)**2+(yy-CC_Webb_tmp.y)**2),axis=None)[n-1]
            grid_nnd_norm[xg,yg] = ((n-1)/(np.pi*dN_norm**2))*(pix_to_parsec2_full/pix_to_parsec2) 
            
            nnd_tmp = []
            for i in range(1,101):
                random_ths = np.random.default_rng().random(len(CC_yso_tmp))
                CC_tmp_th = CC_yso_tmp[(CC_yso_tmp.Prob_PRF>random_ths)]
                dN = np.sort(np.sqrt((xx-CC_tmp_th.x)**2+(yy-CC_tmp_th.y)**2),axis=None)[n-1]
                nnd_tmp.append((n-1)/(np.pi*dN**2))
            grid_nnd[xg,yg] = np.mean(nnd_tmp)*(pix_to_parsec2_full/pix_to_parsec2)  
            grid_nnd_e[xg,yg] = np.std(nnd_tmp)*(pix_to_parsec2_full/pix_to_parsec2)                      
    
    grid_subs = grid_nnd
    grid_e_subs = grid_nnd_e
    grid_to_norm_subs = grid_nnd_norm

    toc = time.perf_counter()
    print(f"Completed contour searching in {(toc - tic)/60:0.2f} minutes!\n\n")

    np.save("sd_grid_nnd_only_substellar", grid_subs)
    np.save("sd_grid_nnd_e_only_substellar", grid_e_subs)
    np.save("sd_grid_norm_nnd_only_substellar",grid_to_norm_subs)
else: 
    grid_subs = np.load("sd_grid_nnd_only_substellar.npy")
    grid_e_subs = np.load("sd_grid_nnd_e_only_substellar.npy")
    grid_to_norm_subs = np.load("sd_grid_norm_nnd_only_substellar.npy")

grid_norm = grid.T/grid_to_norm.T # Normalize, transpose due to inversion of x and y
grid_norm_subs = grid_subs.T/grid_to_norm_subs.T # Normalize, transpose due to inversion of x and y

import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# cont_col = truncate_colormap(plt.get_cmap('gist_heat_r'),0.,.8)
cont_col = 'hot_r'
fig, ax2 = plt.subplots(2,1,figsize=(8,8),sharex=True,sharey=True)
ax2[0].contour(x_col,y_col,cdata,levels,locator=ticker.LogLocator(), cmap=cont_col,alpha=0.5)#,norm=mpl.colors.Normalize()# hot_r works but the yellow (lowest val just above boundary) is too hard to see
vM = max(grid_norm.ravel())
sd = ax2[0].pcolormesh(xgrid,ygrid,grid_norm,vmin=0,vmax=vM,cmap='Greys')
# cbb = plt.colorbar(sd,label="Normalized Surface Density of YSOs",location="right", pad=0.05)

cd = ax2[1].contour(x_col,y_col,cdata,levels,locator=ticker.LogLocator(), cmap=cont_col,alpha=0.5)#,norm=mpl.colors.Normalize()# hot_r works but the yellow (lowest val just above boundary) is too hard to see
ax2[1].pcolormesh(xgrid,ygrid,grid_norm_subs,vmin=0,vmax=vM,cmap='Greys')
# cba = plt.colorbar(cd,label="Column Density",location="right", pad=0.05)
ax2[0].get_xaxis().set_visible(False)
ax2[0].get_yaxis().set_visible(False)
ax2[1].get_xaxis().set_visible(False)
ax2[1].get_yaxis().set_visible(False)
# plt.subplots_adjust(hspace=0)

# ax2.scatter(CC_Webb_Classified.loc[CC_Webb_Classified.Prob_PRF>thresh,'x'],CC_Webb_Classified.loc[CC_Webb_Classified.Prob_PRF>thresh,'y'],s=55,c='maroon',edgecolors='w',linewidth=0.1,marker='*',label=f'YSOs (Prob > {int(thresh*100)}%)')#, transform=tr_webb_wo)
# plt.legend()
plt.axis('off')

scalebar = AnchoredSizeBar(ax2[1].transData,
                        1405, '0.5 pc', 'lower right', 
                        pad=0.5,
                        color='k',
                        frameon=False)#,
                        # size_vertical=1,
                        # fontproperties=fontprops)

ax2[1].add_artist(scalebar)

scalebar = AnchoredSizeBar(ax2[0].transData,
                        1405, '0.5 pc', 'lower right', 
                        pad=0.5,
                        color='k',
                        frameon=False)#,
                        # size_vertical=1,
                        # fontproperties=fontprops)

ax2[0].add_artist(scalebar)

# plt.tight_layout()
ax2[0].set_title('All cYSOs')
ax2[1].set_title('Sub-stellar cYSOs')

fig.tight_layout()
fig.subplots_adjust(right=0.75)
cbar_ax = fig.add_axes([0.78, 0.05, 0.01, 0.9])
fig.colorbar(cd,label="Column Density", cax=cbar_ax)
# fig.subplots_adjust(right=0.87)
cbar_ax_2 = fig.add_axes([0.90, 0.05, 0.01, 0.9])
fig.colorbar(sd,label="Normalized Surface Density", cax=cbar_ax_2)

# fig.tight_layout()
# plt.savefig(f"Figures/{date}/Surf_col_dens_"+date+substel+".png",dpi=300)
plt.savefig(f"Figures/{date}/Surf_col_dens_norm_"+date+substel+".png",dpi=300)
plt.close()

print("Surface/Column density plot saved!")


# ------------------------------------------------------------------------
# Surface/Column Density Figure 2 - per contour level
CC_yso_tmp = CC_Webb_Classified[(CC_Webb_Classified.Class_PRF==0)].copy()#&

mask_cd = ~np.isnan(col_dens_dat.ravel())

tic = time.perf_counter()
if rerun_contours == 'y':
    # Collect average surface density and column density from within each contour
    pts = []
    for l in range(len(levels)):
        # if l+1 < len(levels):
        #     mask = (col_dens_dat>(levels[l]))&((col_dens_dat<levels[l+1])) 
        #     # mask2 = (col_dens[0].data>(levels[l]))&((col_dens[0].data<levels[l+1])) 
        #     mask2 = CC_yso_tmp[(CC_yso_tmp['N(H2)']>(levels[l]))&(CC_yso_tmp['N(H2)']<levels[l+1])].index
        # else: 
        mask = (col_dens_dat>(levels[l]))
        mask2 = CC_yso_tmp[(CC_yso_tmp['N(H2)']>(levels[l]))].index

        if calc_prf_sd.lower()=='y':
            nps2 = len(CC_yso_tmp.loc[mask2])
        else: nps2 = np.load("N_PS_PRF_"+date+".npy")[l]

        nps = np.sum(grid.T[mask])
        nps_e = np.sum(grid_e.T[mask])
        area = len(mask[mask==True])/pix_to_parsec2 # Num px / (px/pc) = pc
        mgas = np.sum(col_dens_dat[mask])*cm2_to_pix*nh2_to_m_gas # Sum of all column densities in all pixels, converted from gas 
        pts.append((nps,nps_e,nps2,mgas,area))

    pts = np.array(pts)

    toc = time.perf_counter()
    print(f"Completed contour searching in {(toc - tic)/60:0.2f} minutes!\n\n")
    N_PS = pts.T[0]
    N_PS_E = pts.T[1]
    N_PS_PRF = pts.T[2]
    M_GAS = pts.T[3]
    A = pts.T[4]

    np.save(filepath+'N_PS_SD_'+substel+date,N_PS)
    np.save(filepath+'N_PS_SD_E_'+substel+date,N_PS_E)
    np.save(filepath+'N_PS_PRF_'+substel+date,N_PS_PRF)
    np.save(filepath+'M_GAS_'+substel+date,M_GAS)
    np.save(filepath+'A_'+substel+date,A)

else: 
    N_PS = np.load(filepath+"N_PS_SD_"+substel+date+".npy")
    N_PS_E = np.load(filepath+"N_PS_SD_E_"+substel+date+".npy")
    N_PS_PRF = np.load(filepath+"N_PS_PRF_"+substel+date+".npy")
    M_GAS = np.load(filepath+"M_GAS_"+substel+date+".npy")
    A = np.load(filepath+"A_"+substel+date+".npy")
if keep_substel=='y':
    M_PS = 0.5 * (1-frac_substel) + 0.05 * (frac_substel)
else:
    M_PS = 0.5
M_PS_E = 0.1
t_PS = 2
t_PS_E = 1
sd_to_sfr = M_PS/t_PS

SIG_GAS = M_GAS/A
SFR = N_PS*sd_to_sfr
SFR_PRF = N_PS_PRF*sd_to_sfr
SFR_E = (N_PS_E*sd_to_sfr)#(1/(A*t_PS)) * np.sqrt((M_PS*N_PS_E)**2+(N_PS*M_PS_E)**2+(N_PS*M_PS*t_PS_E/t_PS)**2)#N_PS_E*sd_to_sfr
SFR_PRF_E =  np.sqrt(N_PS_PRF*sd_to_sfr)#(1/(A*t_PS)) * np.sqrt((M_PS*np.sqrt(N_PS_PRF))**2+(N_PS_PRF*M_PS_E)**2+(N_PS_PRF*M_PS*t_PS_E/t_PS)**2)
SIG_SFR = SFR/A
SIG_SFR_E = SFR_E/A
SIG_SFR_PRF = SFR_PRF/A
SIG_SFR_PRF_E = SFR_PRF_E/A
RHO = (3*np.sqrt(np.pi)*M_GAS/(4*(A**1.5)))*2e30*((3.24078e-17)**3)
T_FF = np.sqrt(3*np.pi/(32*(6.6743e-11)*RHO))/(3.15576e13)


fig, axs = plt.subplots(1,1,sharey=True,figsize=(6,6))
# axs[0].scatter(np.log10(col_dens_dat.ravel()*cm2_to_pix*nh2_to_m_gas/(1/pix_to_parsec2)),np.log10(grid.T.ravel()*sd_to_sfr/(1/pix_to_parsec2)),c=rf_col)
p5 = axs.fill_between(np.log10(SIG_GAS),2*np.log10(SIG_GAS)-4.11-0.3,2*np.log10(SIG_GAS)-4.11+0.3,alpha=0.3)
p6, = axs.plot(np.log10(SIG_GAS),2*np.log10(SIG_GAS)-4.11,ls=':',alpha=0.6,label='Pokhrel et al. 2021 relation')
p1, _, _  = plt.errorbar(np.log10(SIG_GAS),np.log10(SIG_SFR),yerr=np.log10(SIG_SFR_E),ls='None',mec='k',marker='o',c=prf_col,label = 'Via Surface Density')#,xerr=np.log10(0.1*SIG_GAS)
p2, _, _  = plt.errorbar(np.log10(SIG_GAS),np.log10(SIG_SFR_PRF),yerr=np.log10(SIG_SFR_PRF_E),ls='None',mec='k',marker='s',c=prf_col, label='Via YSO Count')
axs.set_ylabel('$\log \Sigma_{\mathrm{SFR}}~ \mathrm{M_\odot/Myr}/\mathrm{pc}^2$')
axs.set_xlabel('$\log \Sigma_{gas}$ $\mathrm{M_\odot/pc}^2$')
lin_fit, cova = np.polyfit(np.log10(SIG_GAS)[~np.isinf(np.log10(SIG_SFR))][np.log10(SIG_GAS)[~np.isinf(np.log10(SIG_SFR))]<2.3], np.log10(SIG_SFR)[~np.isinf(np.log10(SIG_SFR))][np.log10(SIG_GAS)[~np.isinf(np.log10(SIG_SFR))]<2.3], 1, cov=True)
p3, = axs.plot(np.log10(SIG_GAS),lin_fit[0]*np.array(np.log10(SIG_GAS))+lin_fit[1],c=prf_col,label="%.2f" %lin_fit[0]+' * x + '+"%.2f" %lin_fit[1]+" Via Surface Density")
print("%.2f" %lin_fit[0]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[0]+' * x + '+"%.2f" %lin_fit[1]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[1]+" Via Surface Density")
lin_fit, cova = np.polyfit(np.log10(SIG_GAS[~np.isinf(np.log10(SIG_SFR_PRF))]), np.log10(SIG_SFR_PRF[~np.isinf(np.log10(SIG_SFR_PRF))]), 1, cov=True)
p4, = axs.plot(np.log10(SIG_GAS),lin_fit[0]*np.array(np.log10(SIG_GAS))+lin_fit[1],ls='--',c=prf_col,label="%.2f" %lin_fit[0]+' * x + '+"%.2f" %lin_fit[1]+" Via YSO Count")
print("%.2f" %lin_fit[0]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[0]+' * x + '+"%.2f" %lin_fit[1]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[1]+" Via YSO Count")
axs.legend([(p1, p3), (p2, p4), (p5, p6)], ['Surface Density', 'YSO Count', 'Pokhrel et al. 2021'])


fig.tight_layout()

plt.savefig(f'Figures/{date}/surf_vs_col_dens_'+date+substel+'.png',dpi=300)
plt.close()
print("Surface/Column density plot 2 saved!")
print(substel+" results")
print("Total star formation rate for region (SD):", "%.7f"%(SFR[0]/1e6), "or ", str(N_PS[0]), "\pm", str(np.sqrt((N_PS_E[0]**2))), "total stars in region.")#, "$\pm$", "%.3f"%np.nanstd(SIG_SFR),"M$_\odot$/yr")
print("Total star formation rate for region (PRF):", "%.7f"%(float(SFR_PRF[0])/1e6), "or ", str(N_PS_PRF[0]), "\pm", str(np.sqrt((np.sqrt(N_PS_PRF[0])**2))), "total stars in region.")#, "$\pm$", "%.3f"%np.nanstd(SIG_SFR_PRF),"M$_\odot$/yr")
print("Star formation efficiency (PRF) ε = M*/(M*+Mgas) = ", len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0])*M_PS/((M_GAS[0])+len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0])*M_PS))
print("Star formation efficiency (SD) ε = M*/(M*+Mgas) = ", (N_PS[0])*M_PS/((M_GAS[0])+(N_PS[0])*M_PS))
print("Star formation efficiency (PRF) ε = M*/(Mgas) = ", len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0])*M_PS/((M_GAS[0])))
print("Star formation efficiency (SD) ε = M*/(Mgas) = ", (N_PS[0])*M_PS/((M_GAS[0])))
print("Star formation efficiency error (PRF) ε = M*/(M*+Mgas) = ", np.sqrt(len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0]))*M_PS/((M_GAS[0])+len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0])*M_PS))
print("Star formation efficiency error (SD) ε = M*/(M*+Mgas) = ", (N_PS_E[0])*M_PS/((M_GAS[0])+(N_PS[0])*M_PS))
print("Star formation efficiency error (PRF) ε = M*/(Mgas) = ", np.sqrt(len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0]))*M_PS/((M_GAS[0])))
print("Star formation efficiency error (SD) ε = M*/(Mgas) = ", (N_PS_E[0])*M_PS/((M_GAS[0])))
print("Total Mgas is: ",(M_GAS[0]))
print("Total M* is (Final):", len(CC_Webb_Classified[CC_Webb_Classified.Class_PRF==0])*M_PS)
print("Total M* is (SD):", N_PS[0]*M_PS)
# ------------
fig, axs = plt.subplots(1,1,sharey=True,figsize=(6,6))
# axs[0].scatter(np.log10(col_dens_dat.ravel()*cm2_to_pix*nh2_to_m_gas/(1/pix_to_parsec2)),np.log10(grid.T.ravel()*sd_to_sfr/(1/pix_to_parsec2)),c=rf_col)
p5 = axs.fill_between(np.log10(SIG_GAS/T_FF),2*np.log10(SIG_GAS/T_FF)-4.11-0.3,2*np.log10(SIG_GAS/T_FF)-4.11+0.3,alpha=0.3)
p6, = axs.plot(np.log10(SIG_GAS/T_FF),2*np.log10(SIG_GAS/T_FF)-4.11,ls=':',alpha=0.6,label='Pokhrel et al. 2021 relation')
p1, _, _  = plt.errorbar(np.log10(SIG_GAS/T_FF),np.log10(SIG_SFR),yerr=np.log10(SIG_SFR_E),ls='None',mec='k',marker='o',c=prf_col,label = 'Via Surface Density')#,xerr=np.log10(0.1*SIG_GAS)
p2, _, _  = plt.errorbar(np.log10(SIG_GAS/T_FF),np.log10(SIG_SFR_PRF),yerr=np.log10(SIG_SFR_PRF_E),ls='None',mec='k',marker='s',c=prf_col, label='Via YSO Count')
axs.set_ylabel('$\log \Sigma_{\mathrm{SFR}}~ \mathrm{M_\odot/Myr}/\mathrm{pc}^2$')
axs.set_xlabel('$\log \Sigma_{gas}/t_{ff}$ $\mathrm{M_\odot/pc}^2$')
lin_fit, cova = np.polyfit(np.log10(SIG_GAS/T_FF)[~np.isinf(np.log10(SIG_SFR))][np.log10(SIG_GAS/T_FF)[~np.isinf(np.log10(SIG_SFR))]<2.3], np.log10(SIG_SFR)[~np.isinf(np.log10(SIG_SFR))][np.log10(SIG_GAS/T_FF)[~np.isinf(np.log10(SIG_SFR))]<2.3], 1, cov=True)
p3, = axs.plot(np.log10(SIG_GAS/T_FF),lin_fit[0]*np.array(np.log10(SIG_GAS/T_FF))+lin_fit[1],c=prf_col,label="%.2f" %lin_fit[0]+' * x + '+"%.2f" %lin_fit[1]+" Via Surface Density")
print("%.2f" %lin_fit[0]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[0]+' * x + '+"%.2f" %lin_fit[1]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[1]+" Via Surface Density")
lin_fit, cova = np.polyfit(np.log10(SIG_GAS/T_FF)[~np.isinf(np.log10(SIG_SFR_PRF))], np.log10(SIG_SFR_PRF[~np.isinf(np.log10(SIG_SFR_PRF))]), 1, cov=True)
p4, = axs.plot(np.log10(SIG_GAS/T_FF),lin_fit[0]*np.array(np.log10(SIG_GAS/T_FF))+lin_fit[1],ls='--',c=prf_col,label="%.2f" %lin_fit[0]+' * x + '+"%.2f" %lin_fit[1]+" Via YSO Count")
print("%.2f" %lin_fit[0]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[0]+' * x + '+"%.2f" %lin_fit[1]+'\pm'+"%.2f" %np.sqrt(np.diag(cova))[1]+" Via YSO Count")
axs.legend([(p1, p3), (p2, p4), (p5, p6)], ['Surface Density', 'YSO Count', 'Pokhrel et al. 2021'])

fig.tight_layout()

plt.savefig(f'Figures/{date}/surf_vs_col_dens_tff_'+date+substel+'.png',dpi=300)
plt.close()
print("Surface/Column density plot 3 saved!")

# ------------------------------------------------------------------------
# Surface/Column Density Figure 3 - Pokhrel 2021 Figure 1c
NH2 = levels 

fig, ax1 = plt.subplots()

ax1.set_xlabel('N(H$_2$) contour [cm$^{-2}]$')
ax1.set_ylabel('M$_{gas} [M_\odot]$', color=prf_col)
ax1.scatter(NH2, M_GAS, c=prf_col)
ax1.tick_params(axis='y', labelcolor=prf_col)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('N$_{PS}$', color=rf_col)  # we already handled the x-label with ax1
ax2.scatter(NH2, N_PS, c=rf_col)
ax2.tick_params(axis='y', labelcolor=rf_col)
ax2.set_yscale('log')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'Figures/{date}/cd_trends_'+date+substel+'.png',dpi=300)
plt.close()

# ------------------------------------------
# Feature Importance
print("WARNING: The Feature Importances are from the CC_Mar42024_with_10percent run!! (Only be concerned if you're running this script with new data.)")
ranked_features = pd.read_csv("CC_Mar42024/CC_feature_importances_CC_Mar42024_final_class_with_10_percent_saved.csv")
fcd_cols_organized = ranked_features.Feature.values
fig, ax = plt.subplots()
ax.bar(ranked_features.Feature.values,ranked_features.Importance.values/max(ranked_features.Importance.values))
ax.set_xticklabels(labels=ranked_features.Feature.values,rotation=90)
ax.set_xlabel('Input Features (Least to Most Important)')
ax.set_ylabel('Normalized Importance')
fig.tight_layout()
plt.savefig(f'Figures/{date}/feature_importances_'+date+'.png',dpi=300)
plt.close()

#---------------------------------------------
# T-test
ysos_sp_seds = CC_Webb_Classified.loc[yso_pop_sp_bool]
conts_sp_seds = CC_Webb_Classified.loc[cont_pop_sp_bool]
ysos_seds = CC_Webb_Classified.loc[yso_pop_bool]
conts_bl_seds = CC_Webb_Classified.loc[cont_bl_pop_bool]
conts_hl_seds = CC_Webb_Classified.loc[cont_hl_pop_bool]
conts_ll_seds = CC_Webb_Classified.loc[cont_ll_pop_bool]
subs_seds = CC_Webb_Classified.loc[sub_yso_pop_bool]
fps_seds = CC_Webb_Classified.loc[fp_pop_bool]

tind_stat_yso = [ttest_ind(a=ysos_seds[f].values,b=subs_seds[f].values,equal_var=False,nan_policy='omit') for f in fcd_cols_organized]
tind_stat_cont = [ttest_ind(a=conts_ll_seds[f].values,b=subs_seds[f].values,equal_var=False,nan_policy='omit') for f in fcd_cols_organized]
tind_stat_ysofp = [ttest_ind(a=ysos_seds[f].values,b=fps_seds[f].values,equal_var=False,nan_policy='omit') for f in fcd_cols_organized]
tind_stat_contfp = [ttest_ind(a=conts_bl_seds[f].values,b=fps_seds[f].values,equal_var=False,nan_policy='omit') for f in fcd_cols_organized]
tind_stat_contyso = [ttest_ind(a=conts_hl_seds[f].values,b=ysos_seds[f].values,equal_var=False,nan_policy='omit') for f in fcd_cols_organized]
tind_stat_sp_contyso = [ttest_ind(a=conts_sp_seds[f].values,b=ysos_sp_seds[f].values,equal_var=False,nan_policy='omit') for f in fcd_cols_organized]

tstat_yso = np.array([tind_stat_yso[f].statistic for f in range(len(fcd_cols_organized))])
tstat_cont = np.array([tind_stat_cont[f].statistic for f in range(len(fcd_cols_organized))])
tstat_ysofp = np.array([tind_stat_ysofp[f].statistic for f in range(len(fcd_cols_organized))])
tstat_contfp = np.array([tind_stat_contfp[f].statistic for f in range(len(fcd_cols_organized))])
tstat_contyso = np.array([tind_stat_contyso[f].statistic for f in range(len(fcd_cols_organized))])
tstat_sp_contyso = np.array([tind_stat_sp_contyso[f].statistic for f in range(len(fcd_cols_organized))])

tdf_yso = np.array([tind_stat_yso[f].df for f in range(len(fcd_cols_organized))])
tdf_cont = np.array([tind_stat_cont[f].df for f in range(len(fcd_cols_organized))])
tdf_ysofp = np.array([tind_stat_ysofp[f].df for f in range(len(fcd_cols_organized))])
tdf_contfp = np.array([tind_stat_contfp[f].df for f in range(len(fcd_cols_organized))])
tdf_contyso = np.array([tind_stat_contyso[f].df for f in range(len(fcd_cols_organized))])
tdf_sp_contyso = np.array([tind_stat_sp_contyso[f].df for f in range(len(fcd_cols_organized))])

t_pval_yso = np.array([tind_stat_yso[f].pvalue for f in range(len(fcd_cols_organized))])
t_pval_cont = np.array([tind_stat_cont[f].pvalue for f in range(len(fcd_cols_organized))])
t_pval_ysofp = np.array([tind_stat_ysofp[f].pvalue for f in range(len(fcd_cols_organized))])
t_pval_contfp = np.array([tind_stat_contfp[f].pvalue for f in range(len(fcd_cols_organized))])
t_pval_contyso = np.array([tind_stat_contyso[f].pvalue for f in range(len(fcd_cols_organized))])
t_pval_sp_contyso = np.array([tind_stat_sp_contyso[f].pvalue for f in range(len(fcd_cols_organized))])


fig, ax = plt.subplots(1,1,sharex=True,dpi=300)
ax.plot(fcd_cols_organized,(tstat_yso)/tdf_yso,'--',label='Substellar are YSOs')#; Sum: '+"%.2f"%sum(tstat_yso/tdf_yso))
ax.plot(fcd_cols_organized,(tstat_cont)/tdf_cont,':',label='Substellar are Conts')#; Sum: '+"%.2f"%sum(tstat_cont/tdf_cont))
# axs[0].plot(fcd_cols_organized,abs(tstat_ysofp)/tdf_ysofp,'-.',label='False-Positives are YSOs; Sum: '+"%.2f"%sum(tstat_ysofp/tdf_ysofp))
# axs[0].plot(fcd_cols_organized,abs(tstat_contfp)/tdf_contfp,'-.',label='False-Positives are Conts; Sum: '+"%.2f"%sum(tstat_contfp/tdf_contfp))
ax.plot(fcd_cols_organized,(tstat_contyso)/tdf_contyso,label='YSOs and Conts are Same Population')#; Sum: '+"%.2f"%sum(tstat_contyso/tdf_contyso))
ax.plot(fcd_cols_organized,(tstat_sp_contyso)/tdf_sp_contyso,label='Spitzer YSOs and Conts are Same Population')#; Sum: '+"%.2f"%sum(tstat_sp_contyso/tdf_sp_contyso))
ax.legend()

# axs[1].plot(fcd_cols_organized,t_pval_yso,'--',label='Substellar are YSOs; Mean: '+"%.3f"%np.mean(t_pval_yso))
# axs[1].plot(fcd_cols_organized,t_pval_cont,':',label='Substellar are Conts; Mean: '+"%.3f"%np.mean(t_pval_cont))
# # axs[1].plot(fcd_cols_organized,t_pval_ysofp,'-.',label='False-Positives are YSOs; Mean: '+"%.3f"%np.mean(t_pval_ysofp))
# # axs[1].plot(fcd_cols_organized,t_pval_contfp,'-.',label='False-Positives are Conts; Mean: '+"%32f"%np.mean(t_pval_contfp))
# axs[1].plot(fcd_cols_organized,t_pval_contyso,label='YSOs and Conts are Same Population; Mean: '+"%.3f"%np.mean(t_pval_contyso))
# axs[1].plot(fcd_cols_organized,t_pval_sp_contyso,label='Spitzer YSOs and Conts are Same Population; Mean: '+"%.3f"%np.mean(t_pval_sp_contyso))
# axs[1].set_ylim(-0.05,0.2)
# axs[1].plot([fcd_cols_organized[0],fcd_cols_organized[-1]],[0.05,0.05],label='5%')
# axs[1].legend(loc='upper left')
ax.set_xlabel('Input Features (Least to Most Important)')
ax.set_ylabel('T-Test Statistic')
# axs[1].set_ylabel('T-Test P-Value')

plt.rcParams['font.size'] = 5
ax.set_xticklabels(labels=fcd_cols_organized,rotation=90)
plt.rcParams['font.size'] = 12

fig.subplots_adjust(wspace=0)
fig.tight_layout()
plt.savefig(f'Figures/{date}/subs_pop_det_t-test_{date}.png',dpi=300)

