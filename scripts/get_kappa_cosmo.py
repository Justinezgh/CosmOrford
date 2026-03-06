"""
This script generates convergence maps from the Gower Street simulations
Running this file resquires to clone the following repository:
- https://github.com/sachaguer/UNIONS_forward_model
and to install
- camb
- bornraytrace
- glass
- jax-cosmo
- healpy
- h5py
"""

import os
import time
import h5py

import numpy as np
import healpy as hp
import pandas as pd

import glass.shells

from tqdm import tqdm

from ray_trace import ray_trace
from utils import read_cosmo_params, get_path_lightcone, get_path_redshifts, downgrade_lightcone, apply_random_rotation

import jax_cosmo as jc
import jax.numpy as jnp

from forward_model import weight_map_w_redshift
import healpy as hp
from astropy.table import Table
path_sims = "gowerstreet/extracted_data/"
path_infos = "gowerstreet/info_sims.csv"

nz_hsc = Table.read('gowerstreet/nz.fits')

verbose = True

def get_kappa_gower_street(sim_number=1, nside=512):

    if verbose:
        print(f"[!] Preprocessing the Gower Street simulation {sim_number}...")
        print(f"[!] Reading the cosmological parameters...")

    #Read the cosmological parameters
    start = time.time()
    cosmo_params = read_cosmo_params(path_infos, sim_number)
    if verbose:
        print(f"[!] Done in {(time.time()-start)/60:.2f} min.")

    #Get the path to redshift information
    path_exist, path_redshift = get_path_redshifts(path_sims, sim_number)
    assert path_exist, f"The path to the redshift file {path_redshift} does not exist."

    infos_redshift = pd.read_csv(path_redshift, sep=",")

    overdensity_array = []
    z_bin_edges = []

    nside_intermediate=None

    if verbose:
        print(f"[!] Extracting overdensity maps and redshift edges for the Gower Street simulation {sim_number}...")
        pbar = tqdm(zip(infos_redshift["# Step"], infos_redshift["z_far"]))
    else:
        pbar = zip(infos_redshift["# Step"], infos_redshift["z_far"])

    #Fix a bug for sims 779 to 784 where there are more steps than lightcone
    n_step = len(infos_redshift["# Step"])
    step_start = 1
    get_lightcone = False
    start = time.time()
    for step, z_far in pbar:
        if step_start > n_step - 100: #Check the step to avoid bugs for sims 779 to 784
            get_lightcone = True
        else:
            step_start += 1
        if get_lightcone:
            path_exist, path_lightcone = get_path_lightcone(path_sims, sim_number, step-step_start+1)

            if path_exist:
                lightcone = np.load(path_lightcone)
                density_i = lightcone/np.mean(lightcone) - 1
                del lightcone
                if nside_intermediate is not None:
                    density_i = downgrade_lightcone(density_i, nside_intermediate, verbose=False)
                density_i = downgrade_lightcone(density_i, nside, verbose=False)
                overdensity_array.append(density_i)

                z_bin_edges.append(z_far)
    z_bin_edges.append(0.0)

    z_bin_edges = np.array(z_bin_edges[::-1])
    overdensity_array = np.array(overdensity_array[::-1]) #reverse the array as we read from the larger redshift to the smaller

    if verbose:
        print(f"[!] Done in {(time.time()-start)/60:.2f} min.")
        print("[!] Number of redshift shells:", len(z_bin_edges)-1)
        print("[!] Larger redshift:", z_bin_edges[-1])

    kappa_lensing = ray_trace(overdensity_array, 
                            z_bin_edges, 
                            cosmo_params, 
                            method='bornraytrace', 
                            verbose=verbose)

    
    nzs_s = [jc.redshift.kde_nz(jnp.array(nz_hsc['Z_MID'].astype('float32')),
                            jnp.array(nz_hsc['BIN%d'%i].astype('float32')),
                            bw=0.015)
           for i in range(1,4)]

    z = np.array(jnp.linspace(0., 3., 100))
    dndz = np.array(nzs_s[1](z))

    kappa_bar, _ = weight_map_w_redshift(kappa_lensing, z_bin_edges, (dndz, z), verbose=verbose)

    return kappa_bar
    
if __name__ == "__main__":
    sim_number = 1
    nside = 2048
    kappa_bar = get_kappa_gower_street(sim_number=sim_number, nside=nside)
    np.save(f"gowerstreet/kappas/kappa_gower_street_sim{sim_number}.npy", kappa_bar)
