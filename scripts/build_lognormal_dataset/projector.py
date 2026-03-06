import healpy as hp
from functools import partial
import astropy.units as u


def projector_func(array, patch_shape, patch_center, reso = 2 * u.arcmin.to(u.degree), proj_mode = "cartesian"): 
    """
    Selects a patch of an array defined on the sphere and projects it on a 2D map at the desired resolution.  

    array = 
    patch_shape = 2D array specifying the size of the patch (width, height)
    patch_center = 2D array specifying the center of the patch 
    reso = pixel resolution of the final map (choose this wisely in function of the nside of your array)
    proj_mode = Projection type to apply (cartesian or gnomonic)
    """
    nside = hp.pixelfunc.npix2nside(len(array))
    xsize, ysize = patch_shape
    lonc, latc = patch_center 

    if proj_mode == "cartesian":
        # Longitude and latitude windows
        lon_half = (xsize / 2) * reso # longitude half window
        lat_half = (ysize / 2) * reso # latitude half window
        lonra = [lonc - lon_half, lonc + lon_half] # must be [0, 360]
        latra = [latc - lat_half, latc + lat_half] # must be [-90, 90]
        cart_proj = hp.projector.CartesianProj(
            lonra = lonra, latra = latra, xsize = xsize, ysize = ysize
        )
        array_proj = cart_proj.projmap(array, vec2pix_func=partial(hp.vec2pix, nside))

    elif proj_mode == "gnomonic": # congrats for writing gnomonic correctly
        gno_proj = hp.projector.GnomonicProj(
            rot=[lonc, latc, 0], xsize=xsize, ysize=ysize, reso=reso * u.degree.to(u.arcmin) # this projector expects resolution given in arcmin
          )
        array_proj = gno_proj.projmap(array, vec2pix_func=partial(hp.vec2pix, nside))
    else: 
        raise ValueError("proj_mode specified is not implemented yet or does not exist. Choose between 'cartesian' and 'gnomonic'")
    
    return array_proj





    