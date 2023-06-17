import pandas as pd
import numpy as np

from astropy.io import fits
import astropy.wcs as wcs

from astropy.coordinates import match_coordinates_sky,SkyCoord, Angle
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')


date = 'June132023'
filepath = "/Users/breannacrompvoets/DAOPHOT/NGC3324/RESULTS_MJYSR/"

# Open file
head = ['Index','x','y','f090w','e_f090w','f187n','e_f187n','f200w','e_f200w','f335m','e_f335m','f444w','e_f444w','f470n','e_f470n']
dao = pd.read_csv(filepath+"ngc3324_3_pi.raw", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
dao.set_index('Index',inplace=True)
dao.where(dao!=99.9999, np.nan,inplace=True)
dao.where(dao!=9.9999, np.nan,inplace=True)

# Correct ZPs
filters = ['f090w','f187n','f200w','f335m','f444w','f470n']
head = ['ID', 'x', 'y', 'mag_1','mag_2','mag_3','mag_4','mag_5','mag_6','mag_7','mag_8','mag_9','mag_10','mag_11','mag_12']
zps= []
for filt in filters:

    ap_test_adu = pd.read_csv("/users/breannacrompvoets/DAOPHOT/NGC3324/"+filt+"_zp.ap", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
    ap_test_mjysr = pd.read_csv("~/DAOPHOT/NGC3324/RESULTS_MJYSR/"+filt+"_zp_jy.ap", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
    ap_test_adu.where(ap_test_adu!=99.999, np.nan,inplace=True)
    ap_test_adu.where(ap_test_adu!=-99.999, np.nan,inplace=True)
    ap_test_mjysr.where(ap_test_mjysr!=99.999, np.nan,inplace=True)
    ap_test_mjysr.where(ap_test_mjysr!=-99.999, np.nan,inplace=True)
    ap_test_mjysr.where(ap_test_mjysr!=98.999, np.nan,inplace=True)
    ap_test_mjysr.where(ap_test_mjysr!=-98.999, np.nan,inplace=True)
    hdu = fits.open(f'/users/breannacrompvoets/DAOPHOT/NGC3324/RESULTS_MJYSR/'+filt+'.fits')
    k = ap_test_adu.mag_11-ap_test_mjysr.mag_11#*hdu[0].header['PIXAR_SR']*10**6*np.pi*rads[-2]**2) # flux in Jy is flux in MJy/sr * sr/px * Jy/MJy * area of aperature in pix, conversion absorbed into k
    zps.append(np.nanmean(k))
veg_zp = [26.29,22.37,25.60,23.78,24.30,20.22]
for f, filt in enumerate(filters):
    dao[filt] = veg_zp[f]-(dao[filt]+zps[f])

# Add in colours and deltas
dao_tmp =dao.copy()
dao_tmp.dropna(inplace=True)
for filt in filters:
    dao_tmp = dao_tmp[dao_tmp['e_'+filt]<0.05]
dao["f090w-f444w"] = dao['f090w'] - dao['f444w']
dao["e_f090w-f444w"] = np.sqrt(dao['e_f090w'].values**2+dao['e_f444w'].values**2)
for f, filt in enumerate(filters):
    dao[filt+"-"+filters[f+1]] = dao[filt] - dao[filters[f+1]]
    dao["e_"+filt+"-"+filters[f+1]] = np.sqrt(dao["e_"+filt].values**2 + dao["e_"+filters[f+1]].values**2)
    lin_fit = np.polyfit(dao_tmp['f090w'] - dao_tmp['f444w'], dao_tmp[filt]-dao_tmp[filters[f+1]], 1)
    dao["δ_"+filt+"-"+filters[f+1]] = dao[filt]-dao[filters[f+1]] - (lin_fit[0] * (dao['f090w'] - dao['f444w']) + lin_fit[1])
    dao["e_δ_"+filt+"-"+filters[f+1]] = np.sqrt(dao['e_'+filt].values**2+dao['e_'+filters[f+1]].values**2)
    if f == len(filters)-2:
        break

dao["(f090w-f200w)-(f200w-f444w)"] = dao['f090w']-2*dao['f200w']+dao['f444w']
dao["e_δ_(f090w-f200w)-(f200w-f444w)"] = np.sqrt(dao['e_f090w'].values**2+2*dao['e_f200w'].values**2+dao['e_f444w'].values**2)
lin_fit = np.polyfit(dao_tmp['f090w'] - dao_tmp['f444w'], dao_tmp['f090w']-2*dao_tmp['f200w']+dao_tmp['f444w'], 1)
dao["δ_(f090w-f200w)-(f200w-f444w)"] = dao['f090w']-2*dao['f200w']+dao['f444w'] - (lin_fit[0] * (dao['f090w'] - dao['f444w']) + lin_fit[1])
dao["e_δ_(f090w-f200w)-(f200w-f444w)"] = np.sqrt(dao['e_f090w'].values**2+2*dao['e_f200w'].values**2+dao['e_f444w'].values**2)


# Get RA/DEC match
hdu = fits.open(filepath+"f090w.fits")
w_jwst = wcs.WCS(hdu[0].header)

SkyCoords_tmp_dao = w_jwst.pixel_to_world(dao.x,dao.y)
dao['RA'] = SkyCoords_tmp_dao.ra/u.deg
dao['DEC'] = SkyCoords_tmp_dao.dec/u.deg
dao.RA = [float(dao['RA'].values[i]) for i in range(0,len(dao))]
dao.DEC = [float(dao['DEC'].values[i]) for i in range(0,len(dao))]

dao.to_csv(f'DAOPHOT_Catalog_{date}.csv')

# Write to xml for CARTA viewing
from astropy.table import Table
unts = {'x':'pix','y':'pix','f200w':u.mag,'e_f200w':u.mag,'f090w':u.mag,'e_f090w':u.mag,\
    'f187n':u.mag,'e_f187n':u.mag,'f335m':u.mag,'e_f335m':u.mag,'f444w':u.mag,'e_f444w':u.mag,\
        'f470n':u.mag,'e_f470n':u.mag,'RA':u.deg,'DEC':u.deg}
dao_tab = Table.from_pandas(dao,units=unts)
from astropy.io.votable import from_table, writeto
dao_votab = from_table(dao_tab)

writeto(dao_votab, filepath+f"DAOPHOT_{date}.xml")

# Match to Spitzer

# Add SPICY Predictions
# spit2m_cat = pd.read_csv('../Archive/Phase_4__2MASS_UpperLim_Classification/Scripts/NGC_3324_w_Preds.csv')[['RAJ2000','DEJ2000','mag_IR1','e_mag_IR1','mag_IR2','e_mag_IR2','mag_IR3','e_mag_IR3','mag_IR4','e_mag_IR4']]
spicy_cat = pd.read_csv('/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Webb_PRF_Classification/Archive/SPICY_GaiaEDR3.csv',comment='#')
spicy_cat.dropna(subset=['ra_GaiaEDR3','dec_GaiaEDR3'],inplace=True)
# ALL_cat = pd.read_csv('/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Webb_PRF_Classification/Data/All_YSOs_RADEC.csv')
# IR_cat = ALL_cat[ALL_cat.Survey=='IR (Ohlendorf et al. 2013)']


# SPITZER data
# Match to JWST catalog using astropy match_catalog_sky
# Obtain SkyCoords
j_sky = SkyCoord(dao.RA*u.deg, dao.DEC*u.deg)
s2_sky = SkyCoord(spicy_cat.ra_GaiaEDR3*u.deg, spicy_cat.dec_GaiaEDR3*u.deg)
# Set tolerance - matched objects that are at most this far apart are considered one object
tol = 0.003 #in degrees #max(catalog['size'])
# Match
idx, sep2d, x = match_coordinates_sky(s2_sky, j_sky, nthneighbor=1, storekdtree='kdtree_sky')
sep_constraint = sep2d < tol*u.deg
print("Number of Spitzer sources found:",np.count_nonzero(sep_constraint))
# Make new data frames which contain only the matched data, in the order of it's matching
j_matches = dao.iloc[idx[sep_constraint]]
s2_matches = spicy_cat.iloc[sep_constraint]
# Reset the indices in order to match between two catalogues using pd concat
j_matches.reset_index(drop=True,inplace=True)
s2_matches.reset_index(drop=True,inplace=True)
jwst_spitz_cat = pd.concat([j_matches,s2_matches[['yso_candidate','p1']]],axis=1)

# SPICY predictions
# Match to JWST catalog using astropy match_catalog_sky
# j_sky = SkyCoord(jwst_spitz_cat.RAJ2000*u.deg, jwst_spitz_cat.DEJ2000*u.deg)
# sp_sky = SkyCoord(IR_cat.RA*u.deg, IR_cat.DEC*u.deg)
# sp_sky = SkyCoord(spicy_cat['     RAdeg      DEdeg'].to_numpy(),unit=u.deg)
# idx, sep2d, _ = match_coordinates_sky(sp_sky, j_sky, nthneighbor=1, storekdtree='kdtree_sky')
# sep_constraint = sep2d < tol*u.deg
# print("Number of SPICY sources found:",np.count_nonzero(sep_constraint))
# j_matches = jwst_spitz_cat.iloc[idx[sep_constraint]]
# s2_matches = IR_cat.iloc[sep_constraint]
# j_matches.reset_index(inplace=True)
# s2_matches.reset_index(drop=True,inplace=True)
# This catalogue contains only objects with SPICY detections
# jwst_spitz_spicy_cat = pd.concat([j_matches,s2_matches[[' SPICY','class    ','p1']]],axis=1)


# Match to JWST catalog using astropy match_catalog_sky
# j_sky = SkyCoord(jwst_spitz_cat.RAJ2000*u.deg, jwst_spitz_cat.DEJ2000*u.deg)
# sp_sky = SkyCoord(IR_cat.RA*u.deg, IR_cat.DEC*u.deg)
# idx, sep2d, _ = match_coordinates_sky(sp_sky, j_sky, nthneighbor=1, storekdtree='kdtree_sky')
# sep_constraint = sep2d < tol*u.deg
# jwst_spitz_cat['Class'] = [1]*len(jwst_spitz_cat)
# jwst_spitz_cat.loc[idx[sep_constraint],'Class'] = [0]*len(idx[sep_constraint])
# Connect IR predictions to JWST/Spitzer matched catalogue


# Connect SPICY predictions to JWST/Spitzer matched catalogue
# spicy_df_to_add = pd.DataFrame(data={'SPICY':jwst_spitz_spicy_cat[' SPICY'].values,'SPICY_Class':jwst_spitz_spicy_cat['class    '].values,'SPICY_Prob':jwst_spitz_spicy_cat['p1'].values},index=jwst_spitz_spicy_cat['index'])
# df_to_add = pd.DataFrame()
# df_to_add[spicy_df_to_add.columns] = [[np.nan]*len(spicy_df_to_add.columns)]*len(jwst_spitz_cat)
# df_to_add.SPICY_Prob = [0.0]*len(df_to_add)
# df_to_add.iloc[spicy_df_to_add.index] = spicy_df_to_add
# jwst_spitz_cat[df_to_add.columns] = df_to_add

# Specify true and false class
# jwst_spitz_cat['SPICY_Class_0/1'] = 0
# spicy_ind = np.where(np.isnan(jwst_spitz_cat.SPICY))[0]
# jwst_spitz_cat['SPICY_Class_0/1'].loc[spicy_ind] = 1

# Save Spitzer and JWST matched data
jwst_spitz_cat.to_csv(f"DAOPHOT_Catalog_{date}_IR.csv")