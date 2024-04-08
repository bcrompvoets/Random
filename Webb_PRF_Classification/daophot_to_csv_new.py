import pandas as pd
import numpy as np

import os

from astropy.io import fits
import astropy.wcs as wcs
from astropy.coordinates import match_coordinates_sky,SkyCoord, Angle
import astropy.units as u

import warnings
warnings.filterwarnings('ignore')


date = input("Tag to go with new files? (e.g. CC_Aug312023)")
filepath = input("Where are alf and tfr files stored? ")
tfr_name = input("tfr name? ")
n = int(input("Number of filters expected? "))

filepath_dat_sv = "./"+date#input("Filepath to save to? ")
# Make sure the output directory exists before downloading any data
if not os.path.exists(filepath_dat_sv):
    os.makedirs(filepath_dat_sv)


# Allframe from tfr file
files = np.loadtxt(filepath+tfr_name,usecols=(0),max_rows=n,dtype=str)#["f090w_ADU.alf", "f187n_ADU.alf", "f200w_ADU.alf", "f335m_ADU.alf", "f444w_ADU.alf", "f470n_ADU.alf", "f770w.alf", "f1130w.alf", "f1280w.alf", "f1800w.alf"]
filters = [f.split('.')[0].split('_')[0] for f in files]
tfr = pd.read_csv(filepath+tfr_name,header=None,delim_whitespace=True, skiprows=n+1,names=['Index','x','y']+filters)
dao = tfr[['Index','x','y']].copy()
dao.set_index('Index',inplace=True)

for f in filters:
    dao[f] = np.zeros(len(dao))
    dao['e_'+f] = np.zeros(len(dao))


for i, f in enumerate(files):
    filt = f.split('.')[0].split('_')[0]
    print(filt)
    tmp_csv = pd.read_csv(filepath+f, header=None,delim_whitespace=True, skiprows=3, names=['Index','x','y','mag','e_mag','modal_sky','num_it','chi','sharp'])
    tmp_csv.set_index('Index',inplace=True)
    dao.loc[tmp_csv.index, [filt]] = tmp_csv[["mag"]].values
    dao.loc[tmp_csv.index, ['e_'+filt]] = tmp_csv[["e_mag"]].values

dao.where(dao!=0.0, np.nan,inplace=True)

print(dao.info())

print("Size of catalog: ",len(dao))


# Get RA/DEC match
hdu = fits.open(filepath+files[0].split('.')[0]+'.fits')
w_jwst = wcs.WCS(hdu[0].header)

SkyCoords_tmp_dao = w_jwst.pixel_to_world(dao.x,dao.y)
dao['RA'] = [float((SkyCoords_tmp_dao.ra/u.deg)[i]) for i in range(0,len(dao))]
dao['DEC'] = [float((SkyCoords_tmp_dao.dec/u.deg)[i]) for i in range(0,len(dao))]


# Correct ZPs
head = ['ID', 'x', 'y', 'mag_1','mag_2','mag_3','mag_4','mag_5','mag_6','mag_7','mag_8','mag_9','mag_10','mag_11','mag_12']
zps= []
for filt in filters:
    ap_test_adu = pd.read_csv(filepath+filt+"_zp.ap", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
    ap_test_mjysr = pd.read_csv(filepath+filt+"_zp_jy.ap", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
    ap_test_adu.where((99.999-abs(ap_test_adu))>2, np.nan,inplace=True) # +-99.999 now replaced with nan
    ap_test_mjysr.where((99.999-abs(ap_test_mjysr))>2, np.nan,inplace=True) # +-98.999 and +-99.999 now replaced with nan
    hdu = fits.open(filepath+filt+'_mjysr.fits')
    k = ap_test_adu.mag_10-(ap_test_mjysr.mag_10-25+2.5*np.log10(hdu[0].header['PHOTMJSR']))#*hdu[0].header['PIXAR_SR']*10**6*np.pi*rads[-2]**2) # flux in Jy is flux in MJy/sr * sr/px * Jy/MJy * area of aperature in pix, conversion absorbed into k
    zps.append(np.nanmean(k))
print(zps)

veg_zp = pd.read_csv("zero-points_calculated_from_cats_Feb122024.csv")
for f, filt in enumerate(filters):
    dao[filt] = veg_zp.loc[veg_zp['filter']==filt,'zp_vegmag'].values[0]+(dao[filt]-zps[f])

dao.reset_index(inplace=True)
dao.to_csv(f'{filepath_dat_sv}/DAOPHOT_Catalog_{date}.csv',index=False)
print(dao.info())

# Write to xml for CARTA viewing
from astropy.table import Table
unts = {'x':'pix','y':'pix','RA':u.deg,'DEC':u.deg} | {f:u.mag for f in filters} | {'e_'+f:u.mag for f in filters}
dao_tab = Table.from_pandas(dao,units=unts)
from astropy.io.votable import from_table, writeto
dao_votab = from_table(dao_tab)

writeto(dao_votab, filepath_dat_sv+f"/DAOPHOT_{date}.xml")
# writeto(dao_votab, filepath_dat_sv+f"XML Files/DAOPHOT_{date}.xml")

# Match to IR sources
# tol = 0.0001
# yso_IR = pd.read_csv('IR_YSOs_NGC3324.csv')
# cont_IR = pd.read_csv("IR_Conts_NGC3324.csv")
# IR = pd.concat([yso_IR.copy(), cont_IR.copy()]) #[['RA','DEC','Prob']]
# IR.reset_index(inplace=True)
# IR.where(IR!=99.999, np.nan,inplace=True)

# ind, sep, _ = match_coordinates_sky(SkyCoord(IR.RA,IR.DEC,unit=u.deg),SkyCoord(dao.RA,dao.DEC,unit=u.deg))

# sp_bands = [c for c in IR.columns if c[:3]=='mag' or (c[0]=='d' and c[-1]=='m')]
# print("Number of objects matched to Spitzer data: ",len(ind[sep<tol*u.deg]))
# dao_IR = dao.loc[ind[sep<tol*u.deg]]
# dao_IR['Prob'] = IR.loc[sep<tol*u.deg,'Prob'].values
# dao_IR['Class'] = [1]*len(dao_IR)
# dao_IR.loc[dao_IR.Prob>0.5,'Class'] = 0
# dao_IR['Survey'] = IR.loc[sep<tol*u.deg,'Survey'].values
# dao_IR['SPICY_ID'] = IR.loc[sep<tol*u.deg,'SPICY_ID'].values
# dao_IR[sp_bands] = IR.loc[sep<tol*u.deg,sp_bands].values
# dao_IR = dao_IR.loc[abs(dao_IR.f444w-dao_IR.mag4_5)<0.8]
# print("Number of YSOs matched to Spitzer data: ",len(dao_IR[dao_IR['Class']==0]))
# print("Number of Conts matched to Spitzer data: ",len(dao_IR[dao_IR['Class']==1]))

# # Save Spitzer and JWST matched data
# dao_IR.to_csv(f"DAOPHOT_Catalog_{date}_IR.csv",index=False)


# writeto(from_table(Table.from_pandas(yso_IR,units={'RA':u.deg,'DEC':u.deg})), filepath_dat_sv+f"XML Files/YSOs_IR.xml")
# writeto(from_table(Table.from_pandas(cont_IR,units={'RA':u.deg,'DEC':u.deg})), filepath_dat_sv+f"XML Files/Conts_IR.xml")
# writeto(from_table(Table.from_pandas(dao_IR,units={'RA':u.deg,'DEC':u.deg})), filepath_dat_sv+f"XML Files/IR_Webb_Matched.xml")