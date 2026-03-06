from forward_model import log_normal_forward, correct_kappa
from projector import *
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os 
from get_hsc_redshift_distribution import *
from scipy.stats import norm, uniform
from truncated_mvn import TruncatedMVN
import time 
from tqdm import tqdm
import yaml 


N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))
THIS_WORKER = int(os.getenv("SLURM_ARRAY_TASK_ID", 1)) 

def get_prior_from_str(string, params): 
    """
    Sample a distribution specified by a string and prior parameters.
    Only works for a Gaussian and Uniform priors. 
    """
    if string == "uniform": 
        sample = np.random.uniform(low = params[0], high = params[1], size = (1,))
    if string == "normal": 
        sample = params[0] + params[1] * np.random.randn(1)

    return sample.squeeze()

def main(args): 
    input_dir = args.input_dir # working directory.

    # Empty arrays to store the results
    results = np.empty(shape = (args.num_indep_sims, args.num_patches, 1424, 176), dtype = np.float16) # final convergence WITHOUT applying mask 
    labels = np.empty(shape = (args.num_indep_sims, 4), dtype = np.float32) # (seed, Omega_m, S_8, delta_z) for the num_indep_sims

    # Loading the redshift distribution 
    print("Loading the redshift distributions")
    fits_file = os.path.join(input_dir, "hsc_y3/nz.fits")
    z, dndz = get_redshift_distribution(fits_file)

    # Opening the config.yaml file for later (contains parameters of the priors of the cosmological parameters)
    with open("config.yaml", "r") as f: 
        data = yaml.safe_load(f)
    cosmo_params_prior = data["cosmo_params"]
    redshift_shift_prior = data["redshift_shift"]
    camb_accuracies = {
        "AccuracyBoost": data["camb_accuracy"], 
        "lAccuracyBoost": data["camb_accuracy"], 
        "lSampleBoost": data["camb_accuracy"]
    }

    # Looping over the number of sims
    for i in tqdm(range(args.num_indep_sims)):
        seed = THIS_WORKER * args.num_indep_sims + i # different seed for each sim

        # Sampling the cosmological parameters and the redshift uncertainty 
        Omega_m = get_prior_from_str(cosmo_params_prior["prior"], cosmo_params_prior["prior_params"][0])
        S_8 = get_prior_from_str(cosmo_params_prior["prior"], cosmo_params_prior["prior_params"][1])
        delta_z = get_prior_from_str(redshift_shift_prior["prior"], redshift_shift_prior["prior_params"])
        
        print(f"Running for seed = {seed}, Omega_m = {Omega_m:.2f}, S8 = {S_8:.2f}, delta_z = {delta_z:.2f}")
        # Storing the cosmological parameters in a dict for CAMB
        h = 0.7 
        Omega_bh2 = 0.0224
        cosmo_params = { 
            "little_h": h, 
            "Omega_m": Omega_m, 
            "S_8": S_8, 
            "Omega_b": Omega_bh2 / h ** 2
        }


        print(f"Computing the kappa maps on the sphere | Limber approximation = {args.limber_approx}")
        start = time.time()
        nside = args.nside # (set to 2048 for resolution of 2 arcmin as required by the challenge)

        # Computing the kappa maps on the sphere.
        camb_params, kappa_sphere = log_normal_forward(
            cosmo_params = cosmo_params, 
            z = np.clip((z - delta_z), a_min = 0, a_max = None), # clip needed to make sure we don't have any negative redshifts.
            dndz = dndz, 
            nside = nside,
            lmax = 3 * nside - 1, # max l allowed until aliasing according to the Nyquist-Shannon sampling theorem
            dx = args.dx, 
            limber = args.limber_approx, 
            camb_accuracies = camb_accuracies,
            tolerances = [1e8, 1e8],
            rng = np.random.default_rng(seed)
        )
        end = time.time()
        print(f"Finish {(end-start) / 60:.2f} minutes")

        # Creating patches on the sphere to get kappa maps as images.
        kappa_patches = np.empty(shape = (args.num_patches, 1424, 176))  
        lon_cs = np.random.uniform(low = 0, high = 360, size = (args.num_patches,))
        lat_cs = np.random.uniform(low = -90, high = 90, size = (args.num_patches,))

        assert len(lon_cs) == args.num_patches
        for j, patch_center in enumerate(zip(lon_cs, lat_cs)): 
            kappa_proj = projector_func(
                correct_kappa(kappa_sphere),
                patch_center = patch_center,
                patch_shape = (1424, 176), # (width, height)
                reso = args.reso * u.arcmin.to(u.degree),
                proj_mode = "gnomonic" # or 'cartesian' 
            )
            kappa_patches[j] = kappa_proj.T # (back to 'matrix' format)

        # Saving training and labels data
        results[i] = kappa_patches
        labels[i] = [seed, Omega_m, S_8, delta_z]

        if args.debug_mode:
            break 
    
    os.makedirs(args.output_dir, exist_ok = True)
    np.savez(
        args.output_dir + f"/lognormal_sim_{THIS_WORKER}_{args.reso}arcmin_{nside}nside_dx{args.dx}.npz", 
        train=results,
        label=labels
    )



if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()


    parser.add_argument("--output_dir", required = False, type = str)
    parser.add_argument("--input_dir", required = False, type = str, default = None)
    parser.add_argument("--Omega_m", required = False, type = float, help = "If cosmo_params_dir is not given, you must add an Omega_m parameter to give to the script.")
    parser.add_argument("--S_8", required = False, type = float, help = "If cosmo_params_dir is not given, you must add an S_8 parameter to give to the script.")
    parser.add_argument("--limber_approx", required = False, default = True, type = bool)
    parser.add_argument("--num_patches", required = False, default = 5, type = int)
    parser.add_argument("--nside", required = False, default = 2048, type = int)
    parser.add_argument("--reso", required = False, default = 2, type = float, help = "Resolution in arcmin")
    parser.add_argument("--dx", required = False, default = 1000, type = int)
    parser.add_argument("--num_indep_sims", required = False, type = int, help = "Number of independent sims (with different cosmological parameters) to run")
    parser.add_argument("--debug_mode", required = False,  default = False, type = bool)
    args = parser.parse_args()
    main(args)