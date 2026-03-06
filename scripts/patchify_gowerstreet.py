import os
import numpy as np
import healpy as hp
from functools import partial
import argparse
from datasets import Dataset
from utils import read_cosmo_params
import math

def patch_generator(folder, path_infos, nside, reso, patch_shape, lat_limit, N_patches):
    """Generator that yields patches as dictionaries."""
    def random_center(lat_limit=75):
        lon = np.random.uniform(0, 360)
        lat = np.degrees(np.arcsin(
            np.random.uniform(-np.sin(np.radians(lat_limit)), np.sin(np.radians(lat_limit)))
        ))
        return lon, lat

    for i in range(1, 760):  # X goes approximately from 1 to 191
        filename = os.path.join(folder, f"kappa_gower_street_sim{i}.npy")
        if not os.path.exists(filename):
            print(f"File {filename} does not exist, skipping.")
            continue

        # Read cosmological parameters
        cosmo_params = read_cosmo_params(path_infos, i)  # Use `i` as the simulation number
        if cosmo_params is None:
            print(f"Cosmological parameters for simulation {i} not found, skipping.")
            continue

        omega_m = cosmo_params["Omega_m"]
        S8 = cosmo_params["sigma_8"] * math.sqrt(omega_m / 0.3)
        theta = np.array([omega_m.item(), S8.item(), 0., 0., 0.])

        print(f"Processing {filename}")
        kappa_map = np.load(filename)

        for j in range(N_patches):
            lon_center, lat_center = random_center(lat_limit)

            proj = hp.projector.GnomonicProj(
                rot=(lon_center, lat_center, 0),
                xsize=patch_shape[0],
                ysize=patch_shape[1],
                reso=reso,  # arcmin/pixel
            )

            patch = proj.projmap(kappa_map, vec2pix_func=partial(hp.vec2pix, nside))
            yield {"kappa": patch, "theta": theta}

def main():
    parser = argparse.ArgumentParser(description="Extract patches from kappa maps and generate a Hugging Face dataset.")
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Path to the folder containing kappa map files."
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="Path to the folder where the dataset will be stored."
    )
    parser.add_argument(
        "--path_infos", type=str, default="info_sims.csv",
        help="Path to the CSV file containing cosmological parameters (default: info_sims.csv)."
    )
    parser.add_argument(
        "--num_patches", type=int, default=10,
        help="Number of patches to generate per kappa map (default: 10)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    nside = 2048
    reso = 2.0  # arcmin per pixel
    patch_shape = (176, 1424)  # (nx, ny)
    lat_limit = 75  # avoid poles
    N_patches = args.num_patches  # Use the value from the argument

    # Create the dataset from the generator
    dataset = Dataset.from_generator(
        lambda: patch_generator(args.folder, args.path_infos, nside, reso, patch_shape, lat_limit, N_patches)
    )

    # Save the dataset to disk
    dataset.save_to_disk(os.path.join(args.output_folder, "gowerstreet"))
    print(f"Dataset saved to {os.path.join(args.output_folder, 'gowerstreet')}")

if __name__ == "__main__":
    main()
