import pandas as pd
import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
import os


filt = 'f470n'
cols_lst = ['ID','x','y','Mag','Col1','Col2']
dao_l = pd.read_csv(f"~/DAOPHOT/NGC3324/{filt}_ADU.nei", header=None,delim_whitespace=True, skiprows=3, names=cols_lst)

hdu = fits.open("/Users/breannacrompvoets/DAOPHOT/NGC3324/"+filt+"_ADU.fits")

length = 0

inp = open(f"/users/breannacrompvoets/DAOPHOT/NGC3324/{filt}_ADU.nei", "r")
outp = open("/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Webb_PRF_Classification/temp.txt", "w")
# iterate all lines from file
# for line in inp:
for j,line in enumerate(inp):
    # print(line)
    key = 'k'
    if j > 2:
        i = j - 3
        plt.subplots(dpi=70)
        p_s = 50
        plt.imshow(hdu[0].data,vmax=400,origin='lower')
        plt.plot(dao_l.x.iloc[i],dao_l.y.iloc[i],'s',color='red',alpha=0.5)
        plt.xlim(dao_l.x.iloc[i]-p_s,dao_l.x.iloc[i]+p_s)
        plt.ylim(dao_l.y.iloc[i]-p_s,dao_l.y.iloc[i]+p_s)
        plt.show(block=False)
        key = input("Keep (k) or ditch (d)? ")
        plt.close()
    
    if key == 'k' or key == "":
        outp.write(line)
        length +=1
        print(length)
    elif key == 'e':
        outp.close()
        inp.close()
        os.replace('/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Webb_PRF_Classification/temp.txt', f"/users/breannacrompvoets/DAOPHOT/NGC3324/{filt}_ADU.nei")
        break
    elif key == 'd':
        print(j)
        pass
    
    outp.close()
    outp = open("/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Webb_PRF_Classification/temp.txt", "a")
    
    if length == 120:    
        inp.close()
        outp.close()
        os.replace('/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Webb_PRF_Classification/temp.txt', f"/users/breannacrompvoets/DAOPHOT/NGC3324/{filt}_ADU.nei")
        break
        
