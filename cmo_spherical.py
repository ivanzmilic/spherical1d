"""
Rewriting fs_1d_los to be a bit more efficient -- still using a nearest FS
"""

import numpy as np
import matplotlib.pyplot as plt
import lightweaver as lw
import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from astropy.io import fits
import xarray as xr

def compute_intersections(p0, d, radius=1.0):
    # NOTE(cmo): Units of Rs
    # NOTE(im) : Calculates the two intersections between a sphere of radius 1 and a line starting from
    #            the point p0 (x,y,z) and going in the direction d, described by the cosines of angles.
    #            the default usage is going to be x=const=0, z = const = say, 2, and we vary y
    #            d is always going to be (0,0,-1) in this case
    #            of course we can use this for more general purposes

    a = 1.0
    b = 2.0 * (p0 @ d)
    c = p0 @ p0 - radius**2
    delta = b*b - 4.0 * a * c

    if delta < 0:
        return None
    t1 = (-b - np.sqrt(delta)) / (2.0 * a)
    t2 = (-b + np.sqrt(delta)) / (2.0 * a)
    return t1, t2

def sphere_trace_semi_inf(ctx, limbdistance):

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
    outer_intersections = compute_intersections(p0, d)

    num_lambda = ctx.spect.wavelength.shape[0]

    if outer_intersections is None:
        #raise ValueError("Pass a sane mu_out pls")
        I = np.zeros(num_lambda)
        return I,0,0

    # NOTE(cmo): Currently `intersections` represents the intersections with the outermost shell of our atmosphere.
    # Recast 1D atmosphere as set of homogenous spherical shells - chop off the last edge as the lowest shell goes to the core
    z_edges = lw.compute_height_edges(ctx)[:-1]
    # NOTE(cmo): Adjust edges to go "up to" R_sun
    z_edges -= z_edges[0]
    z_edges += const.R_sun.value
    # NOTE(cmo): And convert to normalised [0, 1] radius space
    norm_edges = z_edges / const.R_sun.value
    num_shells = z_edges.shape[0]

    Istart = np.zeros(num_lambda)
    core_ray = False
    # NOTE(cmo): Build the set of shells traversed by a given ray --  this could
    # be done more efficiently, but we're going to do it once per ray
    entering_shell = []
    transition_ts = []
    # Descending through the shells
    for k, e in enumerate(norm_edges):
        shell_int = compute_intersections(p0, d, e)
        if shell_int is None:
            # the previous shell is the deepest one we hit
            break
        entering_shell.append(k)
        transition_ts.append(shell_int[0])
    if entering_shell[-1] != num_shells:
        # Ascend back up through the shells
        for k in range(entering_shell[-1], -1, -1):
            e = norm_edges[k]
            t_in, t_out = compute_intersections(p0, d, e)
            entering_shell.append(k-1)
            transition_ts.append(t_out)
    else:
        # NOTE(cmo): core rays don't ascend
        core_ray = True
        Istart = lw.planck(ctx.atmos.temperature[-1], ctx.spect.wavelength)

    shell_to_sample = np.array(entering_shell[:-1])
    transition_ts = np.array(transition_ts)
    distance_in_shell = transition_ts[1:] - transition_ts[:-1]
    distance_in_shell *= const.R_sun.value

    # NOTE(cmo): Classical back-to-front RT integration
    I = Istart
    total_tau = np.zeros_like(I)
    total_path = 0
    for k, d in zip(shell_to_sample[::-1], distance_in_shell[::-1]):
        chi = ctx.depthData.chi[:, -1, -1, k]
        dtau = chi * d

        total_path += d
        total_tau += dtau

        edt = np.exp(-dtau)
        Sfn = (ctx.depthData.eta[:, -1, -1, k] + ctx.background.sca[:, k] * ctx.spect.J[:, k]) / chi
        I = I * edt + (1.0 - edt) * Sfn

    return I, total_path, total_tau


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
        spec, total_path, total_tau = sphere_trace_semi_inf(susi_ctx, limbdistances[m])
        I[m,:] = spec
        paths[m] = total_path
        taus[m,:] = total_tau

    return I, limbdistances, paths, taus


if __name__=='__main__':

    test_slit = formal_solution_slit()
