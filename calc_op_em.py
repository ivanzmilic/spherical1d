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
from scipy.special import wofz

from contop import continuum_opacity

def planck(wave, T):
    """
    Planck function in cgs units (erg/s/cm^2/sr/Hz)
    wave: wavelength in cm
    T: temperature in K
    """
    nu = const.c.cgs.value / wave
    c1 = 2.0 * const.h.cgs.value / const.c.cgs.value**2
    c2 = const.h.cgs.value / const.k_B.cgs.value
    B = c1*nu**3 / ((np.exp(c2 *nu / T) - 1.0))
    return B

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
    z = vv + damp * 1j
    h = wofz(z).real

    return h/1.7724538509055159 # 1/sqrt(pi)

# The goal is to calculate opacity and emissivity from a 3D cube (one slice of which is a 1D atmosphere, at the time):
# Here we can take any direction (along x,y or z)

def calc_op_em(popfile, murampath, muramid, wavelengths, axis=0, otherids=(0,0), refine =0):

    # axis is the one along which we do the formal solution
    # other_axes are the two perpendicular ones
    # popfile is the population file from promweaver
    # muramid is the muram cube file

    # Load the populations:
    pop = fits.open(popfile)[2].data.transpose(1,0,2,3) # to fix what we messed up earlier
    
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
        v_los = -muramcube.vz.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        ne_los = muramcube.ne.transpose(1,2,0)[otherids[0],:,otherids[1]+150]
        nH_los = (muramcube.Pres.transpose(1,2,0)[otherids[0],:,otherids[1]+150]/(1.38E-16*T_los) - ne_los) * 0.9
        pops = pop[otherids[0],:,:,NZ-otherids[1]-1].transpose(1,0)

    else: return 0 # zeros

    #print (pops.shape)

    pops /= 1E6 # to convert to cm^-3

    from scipy.interpolate import interp1d  
    if (refine):
        # Interpolate to a finer grid:
        T_los = interp1d(np.arange(len(T_los)), T_los, kind='cubic')(np.linspace(0,len(T_los)-1,len(T_los)*refine))
        v_los = interp1d(np.arange(len(v_los)), v_los, kind='cubic')(np.linspace(0,len(v_los)-1,len(v_los)*refine))
        ne_los = interp1d(np.arange(len(ne_los)), ne_los, kind='cubic')(np.linspace(0,len(ne_los)-1,len(ne_los)*refine))
        nH_los = interp1d(np.arange(len(nH_los)), nH_los, kind='cubic')(np.linspace(0,len(nH_los)-1,len(nH_los)*refine))
        pops = interp1d(np.arange(pops.shape[1]), pops, kind='cubic', axis=1)(np.linspace(0,pops.shape[1]-1,pops.shape[1]*refine))

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
    # Fix low and high temperatures:
    Tmin = 4000.0
    Tmax = 1E5
    T_los = np.copy(T_los)
    T_los[T_los<Tmin] = Tmin
    T_los[T_los>Tmax] = Tmax

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
    #print(B_ul/1E9)
    #exit();
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
    # Debug
    #delta_lambda *= 0.0

    vv = (wavelengths[:,None]*1E-7 - llambda0 - delta_lambda[None,:]) / dl_D[None,:]

    #print(vv[:,0])

    # Calculate profiles without the loop:
    phi = fvoigt(a[None,:], vv)

    # Finally calculate op and em, without the loop:
    op = (const.h.cgs.value * nu0 / (4 * np.pi)) * (pops[0][None,:] * B_lu - pops[4][None,:] * B_ul) * phi / dnu_D
    #em = op * planck(393.36E-7, T_los)[None,:]
    
    em = (const.h.cgs.value * nu0 / (4 * np.pi)) * pops[4][None,:] * A_ul * phi / dnu_D
    
    # Stemp:
    #Stemp = pops[4] * A_ul / pops[0] / B_lu
    #print (Stemp[::20])
    opc = continuum_opacity(wavelengths[0,None], T_los, ne_los*1E6, nH_los*1E6)/1E2 # in cm^-1
    emc = opc * planck(wavelengths[0]*1E-7, T_los)
    op += opc[None,:]
    em += emc[None,:]
    
    #print(op[0], opc)
    #exit();
    
    return op, em

def simple_formal_solution(op, em, ds):

    dtau = op[:,:] * ds
    tau = np.cumsum(dtau, axis=1)
    Sfn = em / op
    #print (Sfn[300,::10])
    #print(op[300])
    #print(em[300])    
    #exit();
    transmission = np.exp(-tau)
    smalltau = np.where(tau<1E-2)
    transmission[smalltau] = 1.0 - tau[smalltau] + 0.5 * tau[smalltau]**2 - (1.0/6.0) * tau[smalltau]**3
    
    local_contribution = (1.0 - transmission) * Sfn
    local_contribution[smalltau] = dtau[smalltau] * Sfn[smalltau] * (1.0 - 0.5 * dtau[smalltau] + (1.0/6.0) * dtau[smalltau]**2 - (1.0/24.0) * dtau[smalltau]**3)
    # Now integrate over z (axis=1):
    outgoing_contribution = local_contribution * transmission
    contribution_function_noS = transmission * op
    I = np.sum(outgoing_contribution, axis=1)
    return I, tau[:,-1], contribution_function_noS



if __name__=='__main__':
   
   pops_file= sys.argv[1]
   path_to_muram = sys.argv[2]
   snapshot_id = int(sys.argv[3])

   #wavelengths = np.linspace(392.81432, 394.77057, 1912)
   # Smaller range for testing:
   wavelengths = np.linspace(393.06, 393.66, 601)

   #print(continuum_opacity(393.6, 6000, 1E21, 1E23))
   #exit();

   refine = 2
   
   j = 620 # which y slice to take


   op, em = calc_op_em(pops_file, path_to_muram, snapshot_id, wavelengths,axis=0, otherids=(j, 192), refine=refine)

   # Now let's do a simple formal solution:
   ds = 32e5 # in cm, MURaM grid spacing

   I, tau_los, temp = simple_formal_solution(op, em, ds/refine)
   
   kek = fits.PrimaryHDU(I)
   kek.writeto("test_off_limb.fits",overwrite=True)

   # And plot the results, to test:
   plt.figure(figsize=(10,6))
   plt.plot(wavelengths, I)
   plt.savefig("test_off_limb.png")


   # And now the full slit:
   I_slit = np.zeros((256, len(wavelengths)))
   tau_los = np.zeros((256, len(wavelengths)))
   outgoing_contribution = np.zeros([256, len(wavelengths), 768*2])
   for i in tqdm(range(55,256)):
       op, em = calc_op_em(pops_file, path_to_muram, snapshot_id, wavelengths,axis=0, otherids=(j, i), refine=refine)
       I_slit[i,:], tau_los[i,:], outgoing_contribution[i,:,:] = simple_formal_solution(op, em, ds/refine)

   kek = fits.PrimaryHDU(I_slit)
   bur = fits.ImageHDU(tau_los)
   bur2 = fits.ImageHDU(outgoing_contribution)
   lol = fits.HDUList([kek, bur, bur2])
   lol.writeto(str(j)+"_test_off_limb_slit.fits",overwrite=True)

   z = np.linspace(0,255,256)*32 # in km



   '''
   # And plot the image:
   plt.figure(figsize=(10,6))
   plt.imshow(I_slit[55:,:]*1E-3, origin='lower', aspect='auto',extent=[wavelengths[0],wavelengths[-1],z[55],z[-1]])
   plt.xlim(wavelengths[0], wavelengths[-1])
   plt.ylim(z[55], z[-1])
   plt.colorbar()
   plt.savefig("test_off_limb_slit.png")

   # And the same for tau
   plt.figure(figsize=(10,6))
   plt.imshow(tau_los[55:,:], origin='lower', aspect='auto',extent=[wavelengths[0],wavelengths[-1],z[55],z[-1]])
   plt.xlim(wavelengths[0], wavelengths[-1])
   plt.ylim(z[55], z[-1])
   plt.colorbar()
   plt.savefig("tau_off_limb_slit.png")

   # And the same for tau
   plt.figure(figsize=(10,6))
   plt.imshow((1.0 - np.exp(-tau_los[55:,:])), origin='lower', aspect='auto',extent=[wavelengths[0],wavelengths[-1],z[55],z[-1]])
   plt.xlim(wavelengths[0], wavelengths[-1])
   plt.ylim(z[55], z[-1])
   plt.colorbar()
   plt.title("$1-e^{-\\tau_\\lambda}$")
   plt.savefig("transmission_off_limb_slit.png")
   '''


