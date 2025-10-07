import numpy as np
import matplotlib.pyplot as plt
#import lightweaver as lw
#import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from matplotlib.colors import LogNorm
from astropy.io import fits
#import xarray as xr
import muram as mio
import sys

def fvoigt(damp, vv):
    """
    Based on:
    Voigt function approximation using torch tensors.
    Based on: https://github.com/aasensio/neural_fields/blob/main/utils.py#L174
    """
    A = [122.607931777104326, 214.382388694706425, 181.928533092181549,
         93.155580458138441, 30.180142196210589, 5.912626209773153,
         0.564189583562615]

    B = [122.60793177387535, 352.730625110963558, 457.334478783897737,
         348.703917719495792, 170.354001821091472, 53.992906912940207,
         10.479857114260399, 1.]

    z = damp - np.abs(vv) * 1j

    Z = ((((((A[6] * z + A[5]) * z + A[4]) * z + A[3]) * z + A[2]) * z + A[1]) * z + A[0]) / \
        (((((((z + B[6]) * z + B[5]) * z + B[4]) * z + B[3]) * z + B[2]) * z + B[1]) * z + B[0])

    h = Z.real
    f = np.sign(vv) * Z.imag * 0.5

    return [h, f]

# The goal is to calculate opacity and emissivity from a 3D cube (one slice of which is a 1D atmosphere, at the time):
# Here we can take any direction (along x,y or z)

def calc_op_em(popfile, murampath, muramid, wavelengths, axis=0, otherids=(0,0)):

    # axis is the one along which we do the formal solution
    # other_axes are the two perpendicular ones
    # popfile is the population file from promweaver
    # muramid is the muram cube file

  
    # Load the populations:
    pop = fits.open(popfile)[2].data
    print (pop.shape)


    # Load the MURaM cube:
    muramcube = mio.MuramSnap(murampath, muramid)
    # Nowe we are coming a bit into the hardcoded territory:
    T_los = 0
    v_los = 0
    nH_los = 0
    ne_los = 0


    if (axis==0):
        T_los = muramcube.Temp.transpose(1,2,0)[:,otherids[0],otherids[1]+150]
        v_los = muramcube.vy.transpose(1,2,0)[:,otherids[0],otherids[1]+150]
        ne_los = muramcube.ne.transpose(1,2,0)[:,otherids[0],otherids[1]+150]
        nH_los = (muramcube.Pres.transpose(1,2,0)[:,otherids[0],otherids[1]+150]/(1.38E-16*T_los) - ne_los) * 0.9
        pops = pop[:,otherids[0],:,otherids[1]].transpose(1,0)

    elif (axis==1):
        T_los = muramcube.Temp.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        v_los = muramcube.vy.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        ne_los = muramcube.ne.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        nH_los = (muramcube.Pres.transpose(1,2,0)[otherids[0],:,otherids[1]+150]/(1.38E-16*T_los) - ne_los) * 0.9
        pops = pop[otherids[0],:,:,otherids[1]].tranpose(1,0)

    else: return 0 # zeros

    op = np.zeros((len(wavelengths), len(T_los)))
    em = np.zeros((len(wavelengths), len(T_los)))
    # Just to check:    
    '''
    print(T_los)
    print(v_los)
    print(ne_los)
    print(nH_los)
    '''
    
    return op, em



if __name__=='__main__':
   
   pops_file= sys.argv[1]
   path_to_muram = sys.argv[2]
   snapshot_id = int(sys.argv[3])

   test = calc_op_em(pops_file, path_to_muram, snapshot_id, [393E-9])
