import pandas as pd

"""
utils.py
Author: Sacha Guerrini

Utility functions to perform the forward model of the UNIONS shear maps.
"""


def read_cosmo_params(path_info, sim):
    """
    Returns a dictionary with the cosmologial parameters used to generate the simulation indexed by sim for the Gower Street simulations.

    Parameters
    ----------
    sim : int
        Index of the simulation.

    Returns
    -------
    dict
        Dictionary with the cosmological parameters.
    """
    info = pd.read_csv(path_info, header=1)
    cosmo_params = {}
    line =  info.loc[info["Serial Number"]==sim]
    cosmo_params["h"] = line["little_h"].values
    cosmo_params["Omega_m"] = line["Omega_m"].values
    cosmo_params["Omega_b"] = line["Omega_b"].values
    cosmo_params["sigma_8"] = line["sigma_8"].values
    cosmo_params["n_s"] = line["n_s"].values
    cosmo_params["w"] = line["w"].values
    cosmo_params["m_nu"] = line["m_nu"].values

    #Compute the value of A_s knowing the value of sigma_8
    # res = minimize_scalar(sigma8_difference, args=(cosmo_params,), bounds=[np.log(1e-11), np.log(2e-8)], tol=1e-10)
    # cosmo_params["A_s"] = np.exp(res.x)
    return cosmo_params

