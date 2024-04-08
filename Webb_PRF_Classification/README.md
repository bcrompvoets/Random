# Webb PRF Classification

## daophot_to_csv.py
This file reads in the .raw file output from daophot, applies zp corrections and marks the datapoints which are also seen in Spitzer/WISE

output: DAOPHOT_Catalog_date.csv (all data), DAOPHOT_Catalog_date_IR.csv (data for objects with clear Spitzer counterparts)


## PRF_RF_mp.py
This file runs PRF and RF models 100 different times to evaluate the random error in the runs. The models are trained on the augmented data, tested on the data with Spitzer counterparts, and then used to evaluate the full data and assign classifications and probabilities.

input: DAOPHOT_Catalog_date.csv (all data), DAOPHOT_Catalog_date_IR.csv (data for objects with clear Spitzer counterparts)

output: CC_Classified_DAOPHOT_date.csv

## PRF_DAOPHOT_IR.py
This file creates all figures used for analysis.

input: CC_Classified_DAOPHOT_date.csv

output: See Figures folder