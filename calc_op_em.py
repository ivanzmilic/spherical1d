import numpy as np
import matplotlib.pyplot as plt
import lightweaver as lw
import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from matplotlib.colors import LogNorm
from astropy.io import fits
import xarray as xr
import muram as mio

# The goal is to calculate opacity and emissivity from a 3D cube (one slice of which is a 1D atmosphere, at the time):
# Here we can take any direction (along x,y or z)

def calc_op_em(axis=0, other_axes=(1,2), popfile, muramid):

    # axis is the one along which we do the formal solution
    # other_axes are the two perpendicular ones
    # popfile is the population file from promweaver
    # muramid is the muram cube file

    # Load the populations:
    pop = np.load(popfile)
    print ()

    return op, em



if __name__=='__main__':

   calc_op_em('~/codes/depp_coeff_plp/', 0)
