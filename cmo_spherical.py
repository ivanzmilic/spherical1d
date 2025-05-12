"""
Rewriting fs_1d_los to be a bit more efficient -- still using a nearest FS
"""

from copy import deepcopy
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

def sphere_trace_semi_inf(ctx, limbdistance, velocity=None):
    """
    ctx
        The Lw Context
    limbdistance
        Limb distance in m
    velocity: array_like [Nshell, 3], optional
        Array of 3-component velocities that will be projected to the los
    """

    # NOTE(im): This traces the sphere in a given mu, where p0 actually changes between the rays
    # written by following the one from CMO (a lot of things are simply identical)
    # limbdistance is in km, compared to the absolute possible top of the atmosphere
    # So, it's NOT The actual limb distance
    dy =  1.0 - limbdistance / const.R_sun.value
    #dy = 1.0 * np.sqrt(1.0 - mu_out**2.0)


    # put the origin somewhere far from the atmosphere:
    p0 = np.array([dy, 0.0, 1.0])

    # mu is always going to be -1 in this formulation:
    dir_vec = np.array([0.0, 0.0, -1.0])
    outer_intersections = compute_intersections(p0, dir_vec)

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
        shell_int = compute_intersections(p0, dir_vec, e)
        if shell_int is None:
            # the previous shell is the deepest one we hit
            break
        entering_shell.append(k)
        transition_ts.append(shell_int[0])
    if entering_shell[-1] != num_shells:
        # Ascend back up through the shells
        for k in range(entering_shell[-1], -1, -1):
            e = norm_edges[k]
            t_in, t_out = compute_intersections(p0, dir_vec, e)
            entering_shell.append(k-1)
            transition_ts.append(t_out)
    else:
        # NOTE(cmo): core rays don't ascend
        core_ray = True
        Istart = lw.planck(ctx.atmos.temperature[-1], ctx.spect.wavelength)

    shell_to_sample = np.array(entering_shell[:-1])
    transition_ts = np.array(transition_ts)
    avg_t_in_shell = 0.5 * (transition_ts[1:] + transition_ts[:-1])
    distance_in_shell = transition_ts[1:] - transition_ts[:-1]
    distance_in_shell *= const.R_sun.value

    # NOTE(cmo): Convert wavelength to velocities
    # Use middle of wavelength range... it's probably narrow enough
    lambda0 = ctx.spect.wavelength[int(ctx.spect.wavelength.shape[0] // 2)]
    velocity_axis = (ctx.spect.wavelength - lambda0) / lambda0 * lw.CLight

    # NOTE(cmo): Classical back-to-front RT integration
    I = Istart
    total_tau = np.zeros_like(I)
    total_path = 0
    for k, d, t_ave in zip(shell_to_sample[::-1], distance_in_shell[::-1], avg_t_in_shell[::-1]):
        vel_mu = p0 + t_ave * dir_vec
        vel_mu /= np.linalg.norm(vel_mu)
        if velocity is not None:
            projected_vel = vel_mu @ velocity[k]
            chi = np.interp(velocity_axis + projected_vel, velocity_axis, ctx.depthData.chi[:, -1, -1, k])
            eta = np.interp(velocity_axis + projected_vel, velocity_axis, ctx.depthData.eta[:, -1, -1, k])
        else:
            chi = ctx.depthData.chi[:, -1, -1, k]
            eta = ctx.depthData.chi[:, -1, -1, k]

        dtau = chi * d

        total_path += d
        total_tau += dtau

        edt = np.exp(-dtau)
        Sfn = (eta + ctx.background.sca[:, k] * ctx.spect.J[:, k]) / chi
        I = I * edt + (1.0 - edt) * Sfn

    return I, total_path, total_tau



mu_grid = np.linspace(0.01, 0.02, 2)

def formal_solution_slit(input_atmos=None, velocity_field=None):


    if input_atmos is None:
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
        atmos = deepcopy(input_atmos)
        atmos.quadrature(5)

        rad_set = lw.RadiativeSet(atomic_models)
        rad_set.set_active('H', 'Ca')
        eq_pops = rad_set.compute_eq_pops(atmos)
        spect = rad_set.compute_wavelength_grid()

        hprd = True
        default_ctx = lw.Context(atmos, spect, eq_pops, Nthreads=6, hprd=True)
        lw.iterate_ctx_se(default_ctx, prd=True, quiet=True)

        # NOTE(cmo): Zero out the velocities (populations remain as if they're
        # present), to allow us to interpolate eta/chi
        atmos.vlos[:] = 0.0
        default_ctx.update_deps()
        default_ctx.formal_sol_gamma_matrices()

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

    if velocity_field is None and input_atmos is not None:
        velocity_field = np.zeros((input_atmos.vlos.shape[0], 3))
        velocity_field[:, 2] = input_atmos.vlos

    num_lambda = len(susi_ctx.spect.wavelength)
    I = np.zeros([num_distances, num_lambda])
    paths = np.zeros(num_distances)
    taus = np.zeros([num_distances, num_lambda])
    # NOTE(cmo): I reversed this so the tqdm plays nicely
    for m in tqdm(range(num_distances-1, -1, -1)):
    #for m in range(0,NX):
        spec, total_path, total_tau = sphere_trace_semi_inf(susi_ctx, limbdistances[m], velocity=velocity_field)
        I[m,:] = spec
        paths[m] = total_path
        taus[m,:] = total_tau

    return I, limbdistances, paths, taus


if __name__=='__main__':

    # vels = np.zeros((82, 3))
    # vels[:, 2] = np.sin(np.linspace(0, 6 * np.pi, 82)) * 30e3
    # test_slit = formal_solution_slit(velocity_field=vels)

    # Now let's load some radyn atmosphere:

    ds = xr.open_dataset("/mnt/c/Users/cmo/OneDrive - University of Glasgow/RadynWaves/radyn_out_vars.nc")

    # t = 24900
    t = 25800

    # Note that RADYN Is CGS, and LW is SI

    atmos1d = lw.Atmosphere.make_1d(scale = lw.ScaleType.Geometric, depthScale = ds.z[t].values * 1E-2, temperature = ds.temperature[t].values, \
        vlos = ds.vz[t].values * 1E-2, vturb = np.ones(300)*2E3, ne = ds.ne[t].values*1E6, \
        nHTot = ds.dens[t].values / (lw.DefaultAtomicAbundance.avgMass * lw.Amu*1E3) * 1E6)

    vels = np.zeros((ds.vz[t].values.shape[0], 3))
    vels[:, 2] = ds.vz[t].values * 1e-2 * 5 # Yes I'm making this bigger
    vels[:, 0] = 5e3 * np.sin(ds.z[t].values / ds.z[t].values.max() * 6 * np.pi)

    test_slit = formal_solution_slit(atmos1d, velocity_field=None)
    test_slit_novel = formal_solution_slit(atmos1d, velocity_field=np.zeros((ds.vz[t].values.shape[0], 3)))
    test_slit_xvel = formal_solution_slit(atmos1d, velocity_field=vels)