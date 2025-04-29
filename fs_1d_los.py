import numpy as np
import matplotlib.pyplot as plt
import lightweaver as lw
import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from matplotlib.colors import LogNorm
from astropy.io import fits
import xarray as xr

# Based on a quick and dirty, but amazing, routine written by CMO @ Freiburg in May 2024
# IM: Tuning it a little bit to my taste and then gonna use with opacities and emissivities output by LW

def compute_intersections(p0, d):
    # NOTE(cmo): Units of Rs
    # NOTE(im) : Calculates the two intersections between a sphere of radius 1 and a line starting from 
    #            the point p0 (x,y,z) and going in the direction d, described by the cosines of angles.
    #            the default usage is going to be x=const=0, z = const = say, 2, and we vary y
    #            d is always going to be (0,0,-1) in this case
    #            of course we can use this for more general purposes

    a = 1.0
    b = 2.0 * (p0 @ d)
    c = p0 @ p0 - 1.0
    delta = b*b - 4.0 * a * c

    if delta < 0:
        return None
    t1 = (-b - np.sqrt(delta)) / (2.0 * a)
    t2 = (-b + np.sqrt(delta)) / (2.0 * a)
    return t1, t2

def sphere_trace_semi_inf(ctx, limbdistance, ds = 50):

    # NOTE(im): This traces the sphere in a given mu, where p0 actually changes between the rays 
    # written by following the one from CMO (a lot of things are simply identical)
    # limbdistance is in km, compared to the absolute possible top of the atmosphere
    # So, it's NOT The actual limb distance
    dy =  1.0 - limbdistance / const.R_sun.value
    #dy = 1.0 * np.sqrt(1.0 - mu_out**2.0)
    

    # put the origin somewhere far from the atmosphere:
    p0 = np.array([0.0, dy, 1.0])

    # mu is always going to be -1 in this formulation:
    d = np.array([0,0, -1.0])
    intersections = compute_intersections(p0, d)

    num_lambda = ctx.spect.wavelength.shape[0]
    
    if intersections is None:
        #raise ValueError("Pass a sane mu_out pls")
        I = np.zeros(num_lambda)
        return I,0,0,0
    
    t1, t2 = intersections
    
    R_core = const.R_sun.value
    # Here I am not sure what should be R_total, R_core, Rsun.... Maybe it does not even matter?
    num_sample_points = int((t2-t1) * R_core / 1E3 // ds)
    #print ("info::number of sample points is: ", num_sample_points)
    
    sample_ts = np.linspace(t1, t2, num_sample_points)   
    sample_points = p0 + d[None, :] * sample_ts[:, None]
    #exit();
    
    # Now let's reduce this to the depth
    centre = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    depth = radius - np.sqrt(np.sum((sample_points - centre)**2, axis=1))

    # NOTE(cmo): Dimensionalise and invert sign of depth
    depth *= -const.R_sun.value # in m
    
    # NOTE(cmo): Shift to align 0 with top of our atmosphere
    depth += ctx.atmos.z[0]
    # NOTE(cmo): Limit depth to maximum DEPTH of our atmosphere. i.e. make mask where it's deeper than the deepest FALC (no point in solving)
    # NOTE(im) : We want to also the reduce the length of the actual intersection in order to do the comparison with p-p rays
    mask = depth < ctx.atmos.z[-1]
    
    # Find the greatest depth index
    if np.any(mask):
        max_depth_index = np.argmax(mask)
        depth[max_depth_index:] = ctx.atmos.z[-1]
        depth = depth[:max_depth_index]
        num_sample_points = max_depth_index
        #print ("info::it's a core ray")
        #print ("info:: max_depth index is, ", num_sample_points)
    else: 
        max_depth_index = -1
        #print ("info:: it's a surface ray")

    
    # ds changed a little bit because we cannot have exact sampling we asked for
    ds = (sample_ts[1] - sample_ts[0]) * const.R_sun.value
    # now this should be the actual, geometrical path along the way:
    stotal = num_sample_points * ds # the actual 
    #print(f"info::step {ds/1e3} km, max depth = {depth.min() / 1e3} km")

    # Allocate opacity and emissivity
    eta = np.zeros((num_lambda, num_sample_points))
    chi = np.zeros((num_lambda, num_sample_points))

    # Interpolate using simple linear interpolation:
    for l in range(ctx.spect.wavelength.shape[0]):
        # np.interp (x, xp, yp); we are also fillping the z and opacity, emissivity
        eta[l, :] = np.interp(depth, ctx.atmos.z[::-1], ctx.depthData.eta[l, 0, 0, ::-1])
        chi[l, :] = np.interp(depth, ctx.atmos.z[::-1], ctx.depthData.chi[l, 0, 0, ::-1])
    
    #print(ctx.atmos.z[0]-depth, chi[539,:])
    #for d in range(len(eta[0])):
    #    print((ctx.atmos.z[0]-depth[d])/1E3, chi[539,d])
    # Simple RT solution:
    dtau = chi[:,:] * ds
    # dtau = 0.5 * (chi[1:] + chi[:-1]) * ds
    tau = np.cumsum(dtau, axis=1)

    Sfn = eta / chi
    transmission = np.exp(-tau)
    local_contribution = (1.0 - np.exp(-dtau)) * Sfn
    outgoing_contribution = local_contribution * transmission
    I = np.sum(outgoing_contribution, axis=1)
    return I, outgoing_contribution, stotal, tau[:,-1]


mu_grid = np.linspace(0.01, 0.02, 2)

def formal_solution_slit(input_atmos=None):

    
    if (input_atmos == None):   
        default_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"], prd=True, Nthreads=6)
        #falc_ctx.conserveCharge=True
        #lw.iterate_ctx_se(falc_ctx, prd=True)
        default_ctx.depthData.fill = True
        default_ctx.formal_sol_gamma_matrices()

    else:
        
        """ This is taken from: https://github.com/Goobley/Promweaver/blob/main/promweaver/compute_bc.py
        What we did is we took compute_falc_bc and we want to replace the model atmospheres:
        Configures and iterates a lightweaver Context with a FALC atmosphere, the
        selected atomic models, and active atoms. This can then be used with a
        DynamicContextPromBcProvider.
        """

    
        atomic_models = pw.default_atomic_models()
        atmos = input_atmos
        atmos.quadrature(5)

        rad_set = lw.RadiativeSet(atomic_models)
        rad_set.set_active('H', 'Ca')
        eq_pops = rad_set.compute_eq_pops(atmos)
        spect = rad_set.compute_wavelength_grid()

        hprd = True
        default_ctx = lw.Context(atmos, spect, eq_pops, Nthreads=6, hprd=True)
        lw.iterate_ctx_se(default_ctx, prd=True, quiet=True)

    atlas_wavegrid = np.linspace(391.0, 396.0, 2001)

    I, susi_ctx = default_ctx.compute_rays(wavelengths=lw.air_to_vac(atlas_wavegrid), mus=1.0, returnCtx=True)
    susi_ctx.depthData.fill = True
    susi_ctx.formal_sol_gamma_matrices()

    kek = fits.PrimaryHDU(I)
    kek.writeto("atlas_disk_center.fits", overwrite=True)

    susi_wavegrid = np.linspace(392.81432, 394.77057, 1912)
    I, susi_ctx = default_ctx.compute_rays(wavelengths=lw.air_to_vac(susi_wavegrid), mus=1.0, returnCtx=True)
    susi_ctx.depthData.fill = True
    susi_ctx.formal_sol_gamma_matrices()

    # Experimentation, to show the point:
    #num_distances = 1000
    #limbdistances = np.arange(num_distances) * 50.0+20.0 # exact zero does not work very well. larger limb distance is closer to the disk center.
    #limbdistances *= 1e3 # convert to m please 
    # Actualy SUSI:
    num_distances = 1557
    limbdistances = (np.arange(num_distances) - 784)*19.25+1.0 # Don't hit exactly 0
    limbdistances *= 1e3
    #limbdistances = limbdistances[783:788]
    num_distances = len(limbdistances)
    
    Rs = const.R_sun.value
    #mus = np.sqrt(1.0 - ((Rs-limbdistances)/Rs)**2.0)
    total_z = susi_ctx.atmos.z[0] - susi_ctx.atmos.z[-1] 

    num_lambda = len(susi_ctx.spect.wavelength)
    I = np.zeros([num_distances, num_lambda])
    paths = np.zeros(num_distances)
    taus = np.zeros([num_distances, num_lambda])
    for m in tqdm(range(0, num_distances)):
    #for m in range(0,NX):
        spec, temp, total_path, total_tau = sphere_trace_semi_inf(susi_ctx, limbdistances[m], 25.0)
        I[m,:] = spec 
        paths[m] = total_path
        taus[m,:] = total_tau

    return I, limbdistances, paths, taus

    #kek = fits.PrimaryHDU(I)
    #kek2 = fits.ImageHDU(limbdistances)
    #bur = fits.ImageHDU(paths)
    #lol = fits.ImageHDU(taus)


    #listerino = fits.HDUList([kek,kek2, bur,lol])
    #listerino.writeto("demonstration.fits", overwrite=True)
    #listerino.writeto("susi_synth.fits", overwrite=True)

if __name__=='__main__':

    #test_slit = formal_solution_slit()

    # Now let's load some radyn atmosphere:

    ds = xr.open_dataset("radyn_out_vars.nc")

    t = 0 

    # Note that RADYN Is CGS, and LW is SI

    atmos1d = lw.Atmosphere.make_1d(scale = lw.ScaleType.Geometric, depthScale = ds.z[t].values * 1E-2, temperature = ds.temperature[t].values, \
        vlos = ds.vz[t].values * 1E-2, vturb = np.ones(300)*2E3, ne = ds.ne[t].values*1E6, \
        nHTot = ds.dens[t].values / (lw.DefaultAtomicAbundance.avgMass * lw.Amu*1E3) * 1E6)

    test_slit = formal_solution_slit(atmos1d)
