import jwst
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import match_coordinates_sky,SkyCoord, Angle
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

date = 'Feb172023'

# Filename to save to
saveto = 'CC_Catalog_'+date

# Paths to NIRCam (n) and MIRI (m) data
path_n = "../../ngc3324/NIRCAM_ALL_"+date+"/JWST/jw02731-o001_t017_nircam_"
path_m = "../../ngc3324/MIRI_ALL_"+date+"/JWST/jw02731-o002_t017_miri_"
n_bands = ['f090w','f187n','f200w','f335m','f444w','f444w-f470n']
m_bands = ['f770w', 'f1130w', 'f1280w', 'f1800w']
nm_bands = ['f090w','f187n','f200w','f335m','f444w','f444w-f470n','f770w', 'f1130w', 'f1280w', 'f1800w']
t = []

# Make catalogue which contains one row for every object detected, adding each band to the end **NOT MATCHED**
for i, band in enumerate(n_bands):
    # Get the tables
    # t.append(Table.read(path+band+"/jw02739-o001_t001_nircam_"+band+"_cat.ecsv")) # M16 - Pillars of Creation
    if band != 'f444w-f470n':
        t.append(Table.read(path_n+"clear-"+band+"/jw02731-o001_t017_nircam_clear-"+band+"_cat.ecsv")) # NGC 3324 - Cosmic Cliffs
    else:
        t.append(Table.read(path_n+band+"/jw02731-o001_t017_nircam_"+band+"_cat.ecsv")) # NGC 3324 - Cosmic Cliffs
    print(len(t[i]))


for i, band in enumerate(m_bands):
    # Get the tables
    t.append(Table.read(path_m+band+"/jw02731-o002_t017_miri_"+band+"_cat.ecsv"))
    print(len(t[i+6]))

# Define the measurement types, aperature types
meas_type = 'vegamag'
meas_type2 = 'flux'
ap_type = 'isophotal'
ap_type2 = 'aper30'
ap_type3 = 'aper50'
ap_type4 = 'aper70'
ap_type5 = 'aper_total'

# Add the size of the objects to the bands
for i, f in enumerate(nm_bands):
    if i < 3:
        t[i]['size'] = np.sqrt((t[i]['semimajor_sigma']**2)+t[i]['semiminor_sigma']**2)*0.031/3600 # Convert pixels to arcseconds, arcseconds to degrees (F090,187,200)
    elif i<6:
        t[i]['size'] = np.sqrt((t[i]['semimajor_sigma']**2)+t[i]['semiminor_sigma']**2)*0.063/3600 # Convert pixels to arcseconds, arcseconds to degrees (F335,444,470) 
    else:
        t[i]['size'] = np.sqrt((t[i]['semimajor_sigma']**2)+t[i]['semiminor_sigma']**2)*0.11/3600 # Convert pixels to arcseconds, arcseconds to degrees (MIRI)
    
    t[i] = t[i].to_pandas()[['sky_centroid.ra', 'sky_centroid.dec','size', ap_type+'_'+meas_type, ap_type+'_'+meas_type+'_err', \
        ap_type2+'_'+meas_type, ap_type2+'_'+meas_type+'_err', ap_type3+'_'+meas_type, ap_type3+'_'+meas_type+'_err',\
            ap_type4+'_'+meas_type, ap_type4+'_'+meas_type+'_err',ap_type5+'_'+meas_type, ap_type5+'_'+meas_type+'_err',\
                ap_type+'_'+meas_type2, ap_type+'_'+meas_type2+'_err', \
        ap_type2+'_'+meas_type2, ap_type2+'_'+meas_type2+'_err', ap_type3+'_'+meas_type2, ap_type3+'_'+meas_type2+'_err',\
            ap_type4+'_'+meas_type2, ap_type4+'_'+meas_type2+'_err',ap_type5+'_'+meas_type2, ap_type5+'_'+meas_type2+'_err']]
    t[i].rename(columns={'sky_centroid.ra': 'RA', 'sky_centroid.dec': 'DEC',ap_type+'_'+meas_type:ap_type+'_'+meas_type+'_'+f, ap_type+'_'+meas_type+'_err': ap_type+'_'+meas_type+'_err_'+f,\
        ap_type2+'_'+meas_type:ap_type2+'_'+meas_type+'_'+f, ap_type2+'_'+meas_type+'_err': ap_type2+'_'+meas_type+'_err_'+f,ap_type3+'_'+meas_type:ap_type3+'_'+meas_type+'_'+f, ap_type3+'_'+meas_type+'_err': ap_type3+'_'+meas_type+'_err_'+f,\
            ap_type4+'_'+meas_type:ap_type4+'_'+meas_type+'_'+f, ap_type4+'_'+meas_type+'_err': ap_type4+'_'+meas_type+'_err_'+f,ap_type5+'_'+meas_type:ap_type5+'_'+meas_type+'_'+f, ap_type5+'_'+meas_type+'_err': ap_type5+'_'+meas_type+'_err_'+f,\
                ap_type+'_'+meas_type2:ap_type+'_'+meas_type2+'_'+f, ap_type+'_'+meas_type2+'_err': ap_type+'_'+meas_type2+'_err_'+f,\
        ap_type2+'_'+meas_type2:ap_type2+'_'+meas_type2+'_'+f, ap_type2+'_'+meas_type2+'_err': ap_type2+'_'+meas_type2+'_err_'+f,ap_type3+'_'+meas_type2:ap_type3+'_'+meas_type2+'_'+f, ap_type3+'_'+meas_type2+'_err': ap_type3+'_'+meas_type2+'_err_'+f,\
            ap_type4+'_'+meas_type2:ap_type4+'_'+meas_type2+'_'+f, ap_type4+'_'+meas_type2+'_err': ap_type4+'_'+meas_type2+'_err_'+f,ap_type5+'_'+meas_type2:ap_type5+'_'+meas_type2+'_'+f, ap_type5+'_'+meas_type2+'_err': ap_type5+'_'+meas_type2+'_err_'+f},inplace=True)

# Make into one table, sort
data_all_filters_NIRCam_MIRI = pd.concat([t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9]],ignore_index=True) # MIRI: ,t[6],t[7],t[8],t[9]
data_all_filters_NIRCam_MIRI.sort_values(ap_type+'_'+meas_type+'_f200w', ascending=True,inplace=True)
data_all_filters_NIRCam_MIRI.reset_index(drop=True,inplace=True)


# Use match coordinates sky
# Any object that isn't in index gets appended to the end of the dataframe, this makes the new catalog
# Use new catalog to match next set of objects
filters = ['f090w', 'f187n','f335m','f444w','f444w-f470n','f770w','f1130w','f1280w','f1800w']

# First look at 2 micron data (most populated band)
j_sky = SkyCoord(data_all_filters_NIRCam_MIRI.dropna(subset='isophotal_vegamag_f200w').RA*u.deg, data_all_filters_NIRCam_MIRI.dropna(subset='isophotal_vegamag_f200w').DEC*u.deg)
j_all = data_all_filters_NIRCam_MIRI.dropna(subset='isophotal_vegamag_f200w') # look only at 2 micron data
j_all = j_all[j_all.columns[~j_all.isnull().all()]] # drop nan rows
j_all.reset_index(drop=True,inplace=True) # Start fresh dataframe
# tol = max(data_all_filters_NIRCam_MIRI['size'])

fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].set_xlabel('RA')
axs[1].set_xlabel('DEC')

for f in filters:
    # Running through filters, make a temporary data frame which holds only objects in that filter
    f_df = data_all_filters_NIRCam_MIRI.dropna(subset='isophotal_vegamag_'+f)
    f_df = f_df[f_df.columns[~f_df.isnull().all()]]
    print("# objects in",f,"filter:",len(f_df))
    add_sky = SkyCoord(f_df.RA*u.deg, f_df.DEC*u.deg) # Collect their sky coords
    idx, sep2d, _ = match_coordinates_sky(add_sky,j_sky) # Find the nearest match
    sep_constraint = sep2d < f_df['size'].values*u.deg #make sure the nearest match is within the size of the object

    j_matches = j_all.iloc[idx[sep_constraint]] # All closest objects within sigma distance from catalog
    j_nonmatches = j_all.drop(idx[sep_constraint],axis=0) # All other objects from catalog
    j_matches.reset_index(drop=True,inplace=True)
    j_nonmatches.reset_index(drop=True,inplace=True)

    f_matches = f_df.iloc[sep_constraint] # All closest objects from filter
    f_nonmatches = f_df.iloc[~sep_constraint] # All other objects from filter
    f_matches.reset_index(drop=True,inplace=True)
    f_nonmatches.reset_index(drop=True,inplace=True)

    # Plot the difference between matched objects
    axs[0].hist(j_matches.RA - f_matches.RA,alpha=0.5,bins=np.arange(-0.0001,0.0001,0.00001),label=f)
    axs[1].hist(j_matches.DEC - f_matches.DEC,alpha=0.5,bins=np.arange(-0.00005,0.00005,0.000005),label=f)

    print('RA offset in band ',f,np.mean(j_matches.RA - f_matches.RA),' and DEC offset is ',np.mean(j_matches.DEC - f_matches.DEC))

    j_nonmatches[f_matches.columns[3:]] = np.nan # Fill the new bands with nan
    f_nonmatches[j_matches.columns[3:]] = np.nan # Fill the new bands with nan
    f_nonmatches['RA'] = f_nonmatches.RA + np.mean(j_matches.RA - f_matches.RA) # Shift the new data RA/DEC so it would match with the 2micron band
    f_nonmatches['DEC'] = f_nonmatches.DEC + np.mean(j_matches.DEC - f_matches.DEC)

    j_all = pd.concat([j_matches,f_matches[f_matches.columns[3:]]],axis=1) # Redefine the overall catalogue to be the matched data
    print('# objects matched from',f,"filter:",len(j_all))
    j_all = pd.concat([j_all,j_nonmatches,f_nonmatches],axis=0) # Add unmatched data to the end
    j_all.reset_index(drop=True,inplace=True) # Make a new index
    j_sky = SkyCoord(j_all.RA*u.deg,j_all.DEC*u.deg) # Redefine new j_sky based on matched catalogue

plt.legend()
plt.savefig("Figures/Location_offsets_NGC3324.png",dpi=300)


# Save catalogue without any targets, remove all spurious detections (only one band with detection)
# catalog = j_all
bands = [idx for idx in j_all.columns if (idx[:3] == 'iso' and idx[10]=='v')] 
cols = np.r_[['RA'],['DEC'],bands]
webb_inp_dropped = j_all.dropna(thresh=2,subset=[idx for idx in bands if idx[-8]!='r' and idx[-13]!='r' and idx[10]=='v'])
webb_inp_dropped.to_csv(saveto+'.csv')
print('Number of rows dropped = ',len(j_all)-len(webb_inp_dropped))
catalog = webb_inp_dropped[cols]



# Add SPICY Predictions
spit2m_cat = pd.read_csv('../Archive/Phase_4__2MASS_UpperLim_Classification/Scripts/NGC_3324_w_Preds.csv')
spicy_cat = pd.read_csv('../Archive/SPICY_YSO_SubClasses.csv',comment='#')

spit2m_cat = spit2m_cat[spit2m_cat.RAJ2000<max(catalog.RA)]
spit2m_cat = spit2m_cat[spit2m_cat.RAJ2000>min(catalog.RA)]
spit2m_cat = spit2m_cat[spit2m_cat.DEJ2000<max(catalog.DEC)]
spit2m_cat = spit2m_cat[spit2m_cat.DEJ2000>min(catalog.DEC)]

# SPITZER data
# Match to JWST catalog using astropy match_catalog_sky
# Obtain SkyCoords
j_sky = SkyCoord(catalog.RA*u.deg, catalog.DEC*u.deg)
s2_sky = SkyCoord(spit2m_cat.RAJ2000*u.deg, spit2m_cat.DEJ2000*u.deg)
# Set tolerance - matched objects that are at most this far apart are considered one object
tol = 0.005 #in degrees #max(catalog['size'])
# Match
idx, sep2d, x = match_coordinates_sky(s2_sky, j_sky, nthneighbor=1, storekdtree='kdtree_sky')
sep_constraint = sep2d < tol*u.deg
print("Number of Spitzer sources found:",np.count_nonzero(sep_constraint))
# Make new data frames which contain only the matched data, in the order of it's matching
j_matches = catalog.iloc[idx[sep_constraint]]
s2_matches = spit2m_cat.iloc[sep_constraint]
# Reset the indices in order to match between two catalogues using pd concat
j_matches.reset_index(drop=True,inplace=True)
s2_matches.reset_index(drop=True,inplace=True)
jwst_spitz_cat = pd.concat([j_matches,s2_matches],axis=1)

# SPICY predictions
# Match to JWST catalog using astropy match_catalog_sky
j_sky = SkyCoord(jwst_spitz_cat.RAJ2000*u.deg, jwst_spitz_cat.DEJ2000*u.deg)
sp_sky = SkyCoord(spicy_cat['     RAdeg      DEdeg'].to_numpy(),unit=u.deg)
idx, sep2d, _ = match_coordinates_sky(sp_sky, j_sky, nthneighbor=1, storekdtree='kdtree_sky')
sep_constraint = sep2d < tol*u.deg
print("Number of SPICY sources found:",np.count_nonzero(sep_constraint))
j_matches = jwst_spitz_cat.iloc[idx[sep_constraint]]
s2_matches = spicy_cat.iloc[sep_constraint]
j_matches.reset_index(inplace=True)
s2_matches.reset_index(drop=True,inplace=True)
# This catalogue contains only objects with SPICY detections
jwst_spitz_spicy_cat = pd.concat([j_matches,s2_matches[[' SPICY','class    ','p1']]],axis=1)

# Connect SPICY predictions to JWST/Spitzer matched catalogue
spicy_df_to_add = pd.DataFrame(data={'SPICY':jwst_spitz_spicy_cat[' SPICY'].values,'SPICY_Class':jwst_spitz_spicy_cat['class    '].values,'SPICY_Prob':jwst_spitz_spicy_cat['p1'].values},index=jwst_spitz_spicy_cat['index'])
df_to_add = pd.DataFrame()
df_to_add[spicy_df_to_add.columns] = [[np.nan]*len(spicy_df_to_add.columns)]*len(jwst_spitz_cat)
df_to_add.SPICY_Prob = [0.0]*len(df_to_add)
df_to_add.iloc[spicy_df_to_add.index] = spicy_df_to_add
jwst_spitz_cat[df_to_add.columns] = df_to_add

# Specify true and false class
jwst_spitz_cat['SPICY_Class_0/1'] = 0
spicy_ind = np.where(np.isnan(jwst_spitz_cat.SPICY))[0]
jwst_spitz_cat['SPICY_Class_0/1'].loc[spicy_ind] = 1

# Save Spitzer and JWST matched data
jwst_spitz_cat.to_csv(saveto+"_SPICY_Preds.csv")