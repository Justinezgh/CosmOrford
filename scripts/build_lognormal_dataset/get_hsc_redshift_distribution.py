from astropy.io import fits
import numpy as np

def get_redshift_distribution(fits_file): 
    hdul = fits.open(fits_file)
    Nz = np.array(hdul[1].data["BIN2"]) # N(z) = probability  
    z_low = np.array(hdul[1].data["Z_LOW"]) # low edge of redshift bin
    z_mid = np.array(hdul[1].data["Z_MID"]) # mid point of redshift bin
    z_high = np.array(hdul[1].data["Z_HIGH"]) # high edge of redshift bin
    return (z_mid, Nz)