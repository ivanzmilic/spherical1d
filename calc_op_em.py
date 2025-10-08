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
import scipy.constants as sc
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
    #f = np.sign(vv) * Z.imag * 0.5

    return h

# The goal is to calculate opacity and emissivity from a 3D cube (one slice of which is a 1D atmosphere, at the time):
# Here we can take any direction (along x,y or z)

def calc_op_em(popfile, murampath, muramid, wavelengths, axis=0, otherids=(0,0)):

    # axis is the one along which we do the formal solution
    # other_axes are the two perpendicular ones
    # popfile is the population file from promweaver
    # muramid is the muram cube file

  
    # Load the populations:
    pop = fits.open(popfile)[2].data
    


    # Load the MURaM cube:
    muramcube = mio.MuramSnap(murampath, muramid)
    # Nowe we are coming a bit into the hardcoded territory:
    T_los = 0
    v_los = 0
    nH_los = 0
    ne_los = 0

    NZ = pop.shape[-1]

    if (axis==0):
        T_los = muramcube.Temp.transpose(1,2,0)[:,otherids[0],otherids[1]+150]
        v_los = -muramcube.vy.transpose(1,2,0)[:,otherids[0],otherids[1]+150]
        ne_los = muramcube.ne.transpose(1,2,0)[:,otherids[0],otherids[1]+150]
        nH_los = (muramcube.Pres.transpose(1,2,0)[:,otherids[0],otherids[1]+150]/(1.38E-16*T_los) - ne_los) * 0.9
        pops = pop[:,otherids[0],:,NZ-otherids[1]-1].transpose(1,0)

    elif (axis==1):
        T_los = muramcube.Temp.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        v_los = -muramcube.vy.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        ne_los = muramcube.ne.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        nH_los = (muramcube.Pres.transpose(1,2,0)[otherids[0],:,otherids[1]+150]/(1.38E-16*T_los) - ne_los) * 0.9
        pops = pop[otherids[0],:,:,NZ-otherids[1]-1].tranpose(1,0)

    else: return 0 # zeros

    print (pops.shape)

    pops /= 1E6 # to convert to cm^-3

    #print(pops)

    op = np.zeros((len(wavelengths), len(T_los)))
    em = np.zeros((len(wavelengths), len(T_los)))
    # Just to check:    
    '''
    print(T_los)
    print(v_los)
    print(ne_los)
    print(nH_los)
    '''

    # Equations for opacity and emissivity:
    # op = (h * nu / 4pi) * (n_l B_lu - n_u B_ul) * phi
    # em = (h * nu / 4pi) * n_u A_ul *
    # where phi is the line profile function (Voigt)
    # for phi we need a and doppler width, recalculated then in frequency units

    # Hard-coded line parameters for now, for Ca II 3933:
    g_l = 2
    g_u = 4
    llambda0 = 393.3663E-7 # in cm
    nu0 = const.c.cgs.value / llambda0
    A_ul = 1.47E8
    B_ul = (const.c.cgs.value**2 / (2 * const.h.cgs.value * nu0**3.0)) * A_ul
    B_lu = (g_u / g_l) * B_ul
    #print (B_lu, B_ul, A_ul)
    gamma = A_ul # natural broadening only
    m_Ca = 40.078 * const.u.cgs.value

    # Doppler velocity:
    dv_D = np.sqrt(2 * const.k_B.cgs.value * T_los / m_Ca)
    # Doppler width in frequency units:
    dl_D = (dv_D / const.c.cgs.value) * llambda0
    dnu_D = (dv_D / const.c.cgs.value) * nu0
    # Damping:
    a = gamma / dnu_D
    # Shifted line center in wavelength units:
    delta_lambda = (v_los / const.c.cgs.value) * llambda0

    vv = (wavelengths[:,None]*1E-7 - llambda0 - delta_lambda[None,:]) / dl_D[None,:]

    print(vv[:,0])

    # Calculate profiles without the loop:
    phi = fvoigt(a[None,:], vv)

    # Finally calculate op and em, without the loop:
    op = (const.h.cgs.value * nu0 / (4 * np.pi)) * (pops[0][None,:] * B_lu - pops[2][None,:] * B_ul) * phi / dnu_D
    em = (const.h.cgs.value * nu0 / (4 * np.pi)) * pops[2][None,:] * A_ul * phi / dnu_D
  
    return op, em



if __name__=='__main__':
   
   pops_file= sys.argv[1]
   path_to_muram = sys.argv[2]
   snapshot_id = int(sys.argv[3])

   wavelengths = np.linspace(392.81432, 394.77057, 1912)

   op, em = calc_op_em(pops_file, path_to_muram, snapshot_id, wavelengths,axis=0, otherids=(256, 192))

   # Now let's do a simple 
   ds = 32e5 # in km

   dtau = op[:,:] * ds
   tau = np.cumsum(dtau, axis=1)
   Sfn = em / op
   transmission = np.exp(-tau)
   local_contribution = (1.0 - np.exp(-dtau)) * Sfn
   outgoing_contribution = local_contribution * transmission
   I = np.sum(outgoing_contribution, axis=1)
   
   kek = fits.PrimaryHDU(I)
   kek.writeto("test_off_limb.fits",overwrite=True)


