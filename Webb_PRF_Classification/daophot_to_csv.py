import pandas as pd
import numpy as np

from astropy.io import fits
import astropy.wcs as wcs

from astropy.coordinates import match_coordinates_sky,SkyCoord, Angle
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')


date = 'June192023'
filepath = "/Users/breannacrompvoets/DAOPHOT/NGC3324/"

# Open file
head = ['Index','x','y','f090w','e_f090w','f187n','e_f187n','f200w','e_f200w','f335m','e_f335m','f444w','e_f444w','f470n','e_f470n']
dao = pd.read_csv(filepath+"ngc3324_pi.raw", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
dao.set_index('Index',inplace=True)
dao.where(dao!=99.9999, np.nan,inplace=True)
dao.where(dao!=9.9999, np.nan,inplace=True)

# Correct ZPs
filters = ['f090w','f187n','f200w','f335m','f444w','f470n']
# head = ['ID', 'x', 'y', 'mag_1','mag_2','mag_3','mag_4','mag_5','mag_6','mag_7','mag_8','mag_9','mag_10','mag_11','mag_12']
# zps= []
# for filt in filters:

#     ap_test_adu = pd.read_csv("/users/breannacrompvoets/DAOPHOT/NGC3324/"+filt+"_zp.ap", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
#     ap_test_mjysr = pd.read_csv("~/DAOPHOT/NGC3324/RESULTS_MJYSR/"+filt+"_zp_jy.ap", header=None,delim_whitespace=True, skiprows=3, names=head).iloc[::2, :]
#     ap_test_adu.where(ap_test_adu!=99.999, np.nan,inplace=True)
#     ap_test_adu.where(ap_test_adu!=-99.999, np.nan,inplace=True)
#     ap_test_mjysr.where(ap_test_mjysr!=99.999, np.nan,inplace=True)
#     ap_test_mjysr.where(ap_test_mjysr!=-99.999, np.nan,inplace=True)
#     ap_test_mjysr.where(ap_test_mjysr!=98.999, np.nan,inplace=True)
#     ap_test_mjysr.where(ap_test_mjysr!=-98.999, np.nan,inplace=True)
#     hdu = fits.open(f'/users/breannacrompvoets/DAOPHOT/NGC3324/RESULTS_MJYSR/'+filt+'.fits')
#     k = ap_test_adu.mag_11-ap_test_mjysr.mag_11#*hdu[0].header['PIXAR_SR']*10**6*np.pi*rads[-2]**2) # flux in Jy is flux in MJy/sr * sr/px * Jy/MJy * area of aperature in pix, conversion absorbed into k
#     zps.append(np.nanmean(k))
# veg_zp = [26.29,22.37,25.60,23.78,24.30,20.22]
# for f, filt in enumerate(filters):
#     dao[filt] = veg_zp[f]-(dao[filt]+zps[f])

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
hdu = fits.open(filepath+"f090w_ADU.fits")
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

# Match to IR sources
yso_IR = pd.read_csv('IR_YSOs_RADEC_NGC3324.csv')
cont_IR = pd.read_csv("IR_Conts_RADEC_NGC3324.csv")
IR = pd.concat([yso_IR[['RA','DEC','Prob']].copy(), cont_IR.copy()])
ind, sep, _ = match_coordinates_sky(SkyCoord(IR.RA,IR.DEC,unit=u.deg),SkyCoord(np.transpose(SkyCoords_tmp_dao),unit=u.deg))
dao_IR = dao.loc[ind[sep<0.001*u.deg]]
dao_IR['Prob'] = IR.loc[sep<0.001*u.deg,'Prob'].values
dao_IR['Class'] = [1]*len(dao_IR)
dao_IR.loc[dao_IR.Prob>0.5,'Class'] = 0

# Save Spitzer and JWST matched data
dao_IR.to_csv(f"DAOPHOT_Catalog_{date}_IR.csv")