import pandas as pd
import numpy as np

sp = pd.read_csv("../Archive/Phase_4__2MASS_UpperLim_Classification/Scripts/VelaCarC_l286.csv",comment='#',delimiter=r"\s+")
sp.reset_index(inplace=True)

sp_bands = [c for c in sp.columns if c[:3]=='mag' or (c[0]=='d' and c[-1]=='m')]
all_bands = [c for c in sp.columns if c[:3]=='mag' or (c[0]=='d' and c[-1]=='m') or c=='ra' or c=='dec']

conts = pd.read_csv("IR_YSOs_RADEC_NGC3324.csv")
conts[sp_bands] = [[np.nan]*len(sp_bands)]*len(conts)

for i in range(0,len(conts)):
    ra_tmp = round(conts.loc[i,'RA'],5)
    dec_tmp = round(conts.loc[i,'DEC'],5)
    mask = (abs(sp.dec-dec_tmp)<0.001) & (abs(sp.ra-ra_tmp)<0.001)
    print("Spitzer:\n",sp.loc[mask,['ra','dec']])
    print("New coords: ", i, ra_tmp,dec_tmp)
    if (len(sp.loc[mask,['ra','dec']]))>1:
        k = int(input("Index number for Spitzer? "))
        print(k)
        conts.loc[i,sp_bands] = sp.loc[k,sp_bands].values
    elif (len(sp.loc[mask,['ra','dec']]))==0:
        print("None here")
    else: 
        k = int(sp.loc[mask,['ra','dec']].index[0])
        print(k)
        conts.loc[i,sp_bands] = sp.loc[k,sp_bands].values

conts.to_csv("IR_YSOs_NGC3324.csv")