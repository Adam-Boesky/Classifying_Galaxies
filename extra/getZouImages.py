"""This file is used to match galaxies from the Zou data (es1.v1.fits) to galaxy images from DESI."""
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning) # to quiet Astropy warnings

# 3rd party
import numpy as np

from astropy.utils.data import download_file
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb
from astropy.table import Table
import pandas as pd

import requests
from astropy.io import fits
from io import BytesIO
from astro_ghost.PS1QueryFunctions import *
from astropy.coordinates import SkyCoord, Angle, Distance
from astropy import units as u

#now taking the catalog from https://zenodo.org/record/6620892
dataPath ='/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/data/'
dat = Table.read(dataPath + '/es1.v1.fits', format='fits')
df = dat.to_pandas()
df_gals = df[df['flag_star'] == 0]
df_gals_lowz = df_gals[df_gals['redshift'] < 1.0]
df_gals_lowz_goodPhot = df_gals_lowz[df_gals_lowz['ngoodband'] > 4]
df_gals_lowz_goodPhot_size = df_gals_lowz_goodPhot[(df_gals_lowz_goodPhot['Mstar_gal'] > 1.e8) & (df_gals_lowz_goodPhot['Mstar_gal'] < 1.e12)]
df_gals_lowz_goodPhot.reset_index(drop=True, inplace=True)

saved = 0
pixscale = 0.262
imgs = []
vals = []

for idx, row in df_gals_lowz_goodPhot.iterrows():
    try:
        url = 'https://www.legacysurvey.org/viewer/fits-cutout?ra=%.5f&dec=%.5f&pixscale=%.3f&layer=ls-dr9&size=100'%(row.RA, row.DEC, pixscale)
        r = requests.get(url)
        df = fits.open(BytesIO(r.content),ignore_missing_simple=True)
        img = np.transpose(df[0].data, [1, 2, 0])
        imgs.append(img)
        saved += 1
        if (saved)%101 == 0:
            print("Saved %i"%saved)
        vals.append([row['Tractor_ID'], row['redshift'], np.log10(row['Mstar_gal']), np.log10(1.e9*row['SFR_gal']), row['zphot_lowlim'], row['zphot_upplim'], row['Mstar_gal_err']/row['Mstar_gal'], row['SFR_gal_err']/row['SFR_gal']])
    except OSError:
        continue
np.savez(dataPath + "ZouCrossMatchedSample.npz", x=np.array(imgs), y=np.array(vals))
print("Saved all images!")
len(imgs)
