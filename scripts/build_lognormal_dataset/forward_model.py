import numpy as np
import camb
from cosmology import Cosmology
import glass
import glass.ext.camb
import time
from scipy.optimize import minimize_scalar
import pandas as pd
from tqdm import tqdm 



def get_sigma8_from_As(log_As, cosmo_params):
    h = cosmo_params["little_h"] 
    Omega_m = cosmo_params["Omega_m"]
    Omega_b = cosmo_params["Omega_b"]
    ns = cosmo_params.get("n_s", 0.965)
    m_nu = cosmo_params.get("m_nu", 0.06)
    w = cosmo_params.get("w", -1)
    sigma_8_target = cosmo_params["sigma_8"]

    if m_nu is not None:
        Omega_nu = (m_nu / 93.14) / h**2
    else: 
        Omega_nu = 0
        
    Omega_c = Omega_m - Omega_b - Omega_nu #- camb_params.omeganu
    camb_params = camb.set_params(
        H0=100 * h,
        omch2=Omega_c * h**2,
        ombh2=Omega_b * h**2,
        ns = ns, 
        mnu = m_nu,
        w = w,
        As = np.exp(log_As),
        WantTransfer = True,
        NonLinear=camb.model.NonLinear_both
    )
    # camb_params.set_accuracy(
    # AccuracyBoost=3.0,
    # lAccuracyBoost=3.0,
    # lSampleBoost=3.0
    # )
    results = camb.get_results(camb_params)
    sigma8 = results.get_sigma8()
    return abs(sigma8 - sigma_8_target)
    
def compute_power_spectrum(x, pixsize, kedge):
    """
    Compute the azimuthally averaged 2D power spectrum of a real-valued 2D field.

    Parameters:
    -----------
    x : 2D numpy array
        Input real-space map (e.g., an image or simulated field).
        Must be a 2D array with shape (N_y, N_x).
    
    pixsize : float
        Physical size of each pixel in the map (e.g., arcmin, Mpc, etc.).
        Units should be consistent with the units used for `kedge`.
    
    kedge : 1D array-like
        Bin edges in wavenumber space (k), used to bin the power spectrum.
        Should be monotonically increasing and cover the k-range of interest.

    Returns:
    --------
    power_k : 1D numpy array
        The average wavenumber in each k bin (excluding the DC bin).
    
    power : 1D numpy array
        The binned, azimuthally averaged power spectrum corresponding to `power_k`.
        Normalized per unit area.
    """

    # Ensure the input array is 2D
    assert x.ndim == 2

    # Compute the 2D FFT of the input map and take its squared magnitude (power spectrum)
    xk = np.fft.rfft2(x)  # Real-to-complex FFT (along last axis)
    xk2 = (xk * xk.conj()).real  # Power spectrum: |FFT|^2

    # Get the shape of the input map
    Nmesh = x.shape

    # Compute the wavenumber grid (k-space)
    k = np.zeros((Nmesh[0], Nmesh[1]//2+1))
    # Square of the frequency in the first axis
    k += np.fft.fftfreq(Nmesh[0], d=pixsize).reshape(-1, 1) ** 2
    # Square of the frequency in the second axis (real FFT)
    k += np.fft.rfftfreq(Nmesh[1], d=pixsize).reshape(1, -1) ** 2
    # Convert from (1/length)^2 to angular frequency in radian units
    k = k ** 0.5 * 2 * np.pi

    # Bin each k value according to the bin edges provided in kedge
    index = np.searchsorted(kedge, k)

    # Bin the power values, number of modes, and wavenumbers
    power = np.bincount(index.flatten(), weights=xk2.flatten())
    Nmode = np.bincount(index.flatten())
    power_k = np.bincount(index.flatten(), weights=k.flatten())

    # Adjust for symmetry in the real FFT: include the mirrored part (excluding Nyquist frequency)
    if Nmesh[1] % 2 == 0:  # Even number of columns
        power += np.bincount(index[...,1:-1].flatten(), weights=xk2[...,1:-1].flatten())
        Nmode += np.bincount(index[...,1:-1].flatten())
        power_k += np.bincount(index[...,1:-1].flatten(), weights=k[...,1:-1].flatten())
    else:  # Odd number of columns
        power += np.bincount(index[...,1:].flatten(), weights=xk2[...,1:].flatten())
        Nmode += np.bincount(index[...,1:].flatten())
        power_k += np.bincount(index[...,1:].flatten(), weights=k[...,1:].flatten())

    # Exclude the first bin (typically corresponds to DC mode)
    power = power[1:len(kedge)]
    Nmode = Nmode[1:len(kedge)]
    power_k = power_k[1:len(kedge)]

    # Average the power and wavenumber in each bin, only where Nmode > 0
    select = Nmode > 0
    power[select] = power[select] / Nmode[select]
    power_k[select] = power_k[select] / Nmode[select]

    # Normalize the power spectrum by the map area
    power *= pixsize ** 2 / Nmesh[0] / Nmesh[1]

    # Return the binned k values and corresponding power spectrum
    return power_k, power


def cosmo_to_camb_params(cosmo_params):
    h = cosmo_params["little_h"] 
    Omega_m = cosmo_params["Omega_m"]
    Omega_b = cosmo_params["Omega_b"]
    ns = cosmo_params.get("n_s", 0.965)
    m_nu = cosmo_params.get("m_nu", 0.06)
    w = cosmo_params.get("w", -1)
    if m_nu is not None:
        Omega_nu = (m_nu / 93.14) / h**2
    else: 
        Omega_nu = 0
    Omega_c = Omega_m - Omega_b - Omega_nu #- camb_params.omeganu
    log_As = minimize_scalar(lambda x: get_sigma8_from_As(x, cosmo_params), bounds=[np.log(1e-11), np.log(2e-8)], tol=1e-11).x
    camb_params = camb.set_params(
        H0=100 * h,
        omch2=Omega_c * h**2,
        ombh2=Omega_b * h**2,
        ns = ns, 
        mnu = m_nu,
        w = w,
        As = np.exp(log_As),
        WantTransfer = True, 
        NonLinear=camb.model.NonLinear_both,
    )
    return camb_params



import warnings
# Creating our own solve_gaussian_spectra func to specify tolerances 
def solve_gaussian_spectra(fields, spectra, tolerances = [1e-5, 1e-5]):
    """
    Solve a sequence of Gaussian angular power spectra.

    After transformation by *fields*, the expected two-point statistics
    should recover *spectra* when using a non-band-limited transform
    [Tessore23]_.

    Parameters
    ----------
    fields
        The fields to be simulated.
    spectra
        The desired angular power spectra of the fields.

    Returns
    -------
        Gaussian angular power spectra for simulation.

    """
    n = len(fields)
    if len(spectra) != n * (n + 1) // 2:
        msg = "mismatch between number of fields and spectra"
        raise ValueError(msg)

    gls = []
    for i, j, cl in glass.enumerate_spectra(spectra):
        if cl.size > 0:
            # transformation pair
            t1, t2 = fields[i], fields[j]

            # set zero-padding of solver to 2N
            pad = 2 * cl.size

            # if the desired monopole is zero, that is most likely
            # and artefact of the theory spectra -- the variance of the
            # matter density in a finite shell is not zero
            # -> set output monopole to zero, which ignores cl[0]
            monopole = 0.0 if cl[0] == 0 else None

            # call solver
            gl, _cl_out, info = glass.grf.solve(
                cl, 
                t1, 
                t2, 
                pad=pad, 
                monopole=monopole, 
                cltol = tolerances[0], 
                gltol = tolerances[1])

            # warn if solver didn't converge
            if info == 0:
                warnings.warn(
                    f"Gaussian spectrum for fields ({i}, {j}) did not converge",
                    stacklevel=2,
                )
        else:
            gl = 0 * cl  # makes a copy of the empty array
        gls.append(gl)
    return gls



def log_normal_forward(cosmo_params, z, dndz, rng, nside = 2048, lmax = 2048 * 3, limber = False, dx = 150, camb_accuracies = None, tolerances = [1e-5, 1e-5]):
    """
    For a set of cosmological parameters (Omega_m, S_8), computes 
    the convergence maps on the sphere.
    """

    try:
        sigma_8_target = cosmo_params["sigma_8"]
    except KeyError: 
        S8 = cosmo_params["S_8"]
        sigma_8_target =  S8 * (0.3 / cosmo_params["Omega_m"]) ** 0.5
        cosmo_params["sigma_8"] = sigma_8_target

    
    # basic parameters of the simulation
    lmax = 3 * nside # maximum before aliasing occurs according to Nyquist-Shannon theorem; 

    # get the cosmology from CAMB
    camb_params = cosmo_to_camb_params(cosmo_params)
    # Setting Camb accuracies
    camb_params.set_accuracy(
        AccuracyBoost = camb_accuracies["AccuracyBoost"], 
        lAccuracyBoost = camb_accuracies["lAccuracyBoost"], 
        lSampleBoost = camb_accuracies["lSampleBoost"]
    )
    cosmo = Cosmology.from_camb(camb_params)

    print("CAMB params initialized") 
    # shells of dx Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, z[0], z[-1], dx=dx)
    print(f"Computing for {zb.shape} shells")

    # linear radial window functions
    shells = glass.linear_windows(zb)
    start = time.time()
    cls = glass.ext.camb.matter_cls(camb_params, lmax, shells, limber = limber)
    end = time.time()
    print(f"Power spectra per shell computed in {(end-start):.2f}")

    start = time.time()
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)
    end = time.time()
    print(f"Discretizing cls computed in {(end-start):.2f}")

    # Initialize LogNormal fields on all shells
    start = time.time()
    fields = glass.lognormal_fields(shells)
    gls = solve_gaussian_spectra(fields, cls, tolerances)
    end = time.time()
    print(f"Solving Gaussian spectra computed in {(end-start):.2f}")

    # Initialize the initial density fluctuations. 
    matter = glass.generate(fields, gls, nside, ncorr = 3, rng = rng)

    # Initialize the object computing the convergence map
    convergence_calculator = glass.MultiPlaneConvergence(cosmo)

    # Partition the galaxies into redshift shells
    ngal = glass.partition(z, dndz, shells) 
    
    kappa_bar = np.zeros(12 * nside**2)

    # main loop to simulate the matter fields iterative
    print("Computing the convergence")
    for i, delta_i in tqdm(enumerate(matter)):
        # add lensing plane from the window function of this shell
        convergence_calculator.add_window(delta_i, shells[i])

        # get convergence field
        kappa_i = convergence_calculator.kappa

        # add to mean fields using the galaxy number density as weight
        kappa_bar += ngal[i] * kappa_i

    # normalise mean fields by the total galaxy number density
    kappa_bar /= ngal.sum()

    return camb_params, kappa_bar


import healpy as hp
def correct_kappa(kappa): 
    # kappa_sphere: healpix map (1D array of length 12 * nside^2)
    nside = hp.get_nside(kappa)
    lmax = 3 * nside - 1
    
    # Compute the spherical harmonic coefficients
    alm_obs = hp.map2alm(kappa, lmax=lmax)
    
    # Get the pixel window function (same lmax)
    pw = hp.pixwin(nside, lmax=lmax)
    
    # Divide each alm by the pixel window func
    l, m = hp.Alm.getlm(lmax)
    alm_true = alm_obs / pw[l]
    
    # Reconstruct the corrected map
    kappa_corrected = hp.alm2map(alm_true, nside=nside, lmax=lmax)
    return kappa_corrected
    
