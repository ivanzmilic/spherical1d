import numpy as np
import matplotlib.pyplot as plt
import lightweaver as lw
import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from matplotlib.colors import LogNorm

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

def sphere_trace_ctx(ctx, mu, num_sample_points=100):
    
    # put the origin somewhere far from the atmosphere:
    
    p0 = np.array([0.0, 0.0, 2.0])
    
    # mu is always going to be -1 but be flexible
    d = np.array([-np.sqrt(1.0 - mu**2), 0.0, -mu])
    intersections = compute_intersections(p0, d)
    if intersections is None:
        raise ValueError("Pass a sane mu pls")
    t1, t2 = intersections
    sample_ts = np.linspace(t1, t2, num_sample_points)
    sample_points = p0 + d[None, :] * sample_ts[:, None]
    centre = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    depth = radius - np.sqrt(np.sum((sample_points - centre)**2, axis=1))
    # NOTE(cmo): Dimensionalise and invert sign of depth
    depth *= -const.R_sun.value
    # NOTE(cmo): Shift to align 0 with top of FALC
    depth += ctx.atmos.z[0]
    # NOTE(cmo): Limit depth to maximum of FALC
    mask = depth < ctx.atmos.z[-1]
    if np.any(mask):
        max_depth_index = np.argmax(mask)
        depth[max_depth_index:] = ctx.atmos.z[-1]

    ds = (sample_ts[1] - sample_ts[0]) * const.R_sun.value
    print(f"Step: {ds/1e3} km, max depth = {depth.min() / 1e3} km")

    eta = np.zeros((ctx.spect.wavelength.shape[0], num_sample_points))
    chi = np.zeros((ctx.spect.wavelength.shape[0], num_sample_points))

    for la in range(ctx.spect.wavelength.shape[0]):
        # eta[la, :] = weno4(depth, ctx.atmos.z, ctx.depthData.eta[la, 0, 0, :])
        # chi[la, :] = weno4(depth, ctx.atmos.z, ctx.depthData.chi[la, 0, 0, :])
        eta[la, :] = np.interp(depth, ctx.atmos.z[::-1], ctx.depthData.eta[la, 0, 0, ::-1])
        chi[la, :] = np.interp(depth, ctx.atmos.z[::-1], ctx.depthData.chi[la, 0, 0, ::-1])
    dtau = chi * ds
    # dtau = 0.5 * (chi[1:] + chi[:-1]) * ds
    tau = np.cumsum(dtau, axis=1) - tau[0]
    Sfn = eta / chi
    transmission = np.exp(-tau)
    local_contribution = (1.0 - np.exp(-dtau)) * Sfn
    outgoing_contribution = local_contribution * transmission
    I = np.sum(outgoing_contribution, axis=1)
    return I, outgoing_contribution, ds

def sphere_trace_semi_inf(mu_out, ds = 50):

    # NOTE(im): This traces the sphere in a given mu, where p0 actually changes
    # written by following the one above  
    # Figure out mu_out, and thus the coordinate of the observing point:
    
    dy = 1.0 * np.sqrt(1.0 - mu_out**2.0)
    # put the origin somewhere far from the atmosphere:
    
    p0 = np.array([0.0, dy, 1.0])

    # mu is always going to be -1 in this formulation:
    d = np.array([0,0, -1.0])
    intersections = compute_intersections(p0, d)
    
    if intersections is None:
        raise ValueError("Pass a sane mu_out pls")
        return 0
    
    t1, t2 = intersections
    print (t1,t2)

    R_core = const.R_sun.value
    num_sample_points = int((t2-t1) * R_core / 1E3 // ds)
    print ("info::number of sample points is: ", num_sample_points)
    
    sample_ts = np.linspace(t1, t2, num_sample_points)   
    sample_points = p0 + d[None, :] * sample_ts[:, None]
    #exit();
    
    # Now let's reduce this to the depth
    centre = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    depth = radius - np.sqrt(np.sum((sample_points - centre)**2, axis=1))



    # NOTE(cmo): Dimensionalise and invert sign of depth
    depth *= -const.R_sun.value

    print(depth)
    return depth
    '''
    # NOTE(cmo): Shift to align 0 with top of FALC
    depth += ctx.atmos.z[0]
    # NOTE(cmo): Limit depth to maximum DEPTH of FALC
    mask = depth < ctx.atmos.z[-1]
    if np.any(mask):
        max_depth_index = np.argmax(mask)
        depth[max_depth_index:] = ctx.atmos.z[-1]

    ds = (sample_ts[1] - sample_ts[0]) * const.R_sun.value
    print(f"Step: {ds/1e3} km, max depth = {depth.min() / 1e3} km")

    eta = np.zeros((ctx.spect.wavelength.shape[0], num_sample_points))
    chi = np.zeros((ctx.spect.wavelength.shape[0], num_sample_points))

    for la in range(ctx.spect.wavelength.shape[0]):
        # eta[la, :] = weno4(depth, ctx.atmos.z, ctx.depthData.eta[la, 0, 0, :])
        # chi[la, :] = weno4(depth, ctx.atmos.z, ctx.depthData.chi[la, 0, 0, :])
        eta[la, :] = np.interp(depth, ctx.atmos.z[::-1], ctx.depthData.eta[la, 0, 0, ::-1])
        chi[la, :] = np.interp(depth, ctx.atmos.z[::-1], ctx.depthData.chi[la, 0, 0, ::-1])
    dtau = chi * ds
    # dtau = 0.5 * (chi[1:] + chi[:-1]) * ds
    tau = np.cumsum(dtau, axis=1) - tau[0]
    Sfn = eta / chi
    transmission = np.exp(-tau)
    local_contribution = (1.0 - np.exp(-dtau)) * Sfn
    outgoing_contribution = local_contribution * transmission
    I = np.sum(outgoing_contribution, axis=1)
    return I, outgoing_contribution, ds'''




mu_grid = np.linspace(0.01, 0.02, 2)

if __name__ == "__main__":

    
    falc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"], Nthreads=6)
    falc_ctx.depthData.fill = True
    falc_ctx.formal_sol_gamma_matrices()

    plane_parallel_tabulate = pw.tabulate_bc(falc_ctx, mu_grid=mu_grid)
    lw_I = plane_parallel_tabulate["I"]

    mu_out = 0.01
    kek = sphere_trace_semi_inf(mu_out, 50)

   
    '''
    spherical_I = np.zeros((falc_ctx.spect.wavelength.shape[0], mu_grid.shape[0]))
    cfn = np.zeros((falc_ctx.spect.wavelength.shape[0], mu_grid.shape[0], 20_000))
    ds = np.zeros(mu_grid.shape[0])
    for mu_idx, mu in enumerate(tqdm(mu_grid)):
        spherical_I[:, mu_idx], cfn[:, mu_idx, :], ds[mu_idx] = sphere_trace_ctx(falc_ctx, mu)

    wave = falc_ctx.spect.wavelength
    wave_edges = np.concatenate((
        [wave[0]],
        0.5 * (wave[1:] + wave[:-1]),
        [wave[-1]]
    ))
    mu_edges = np.concatenate((
        [mu_grid[0]],
        0.5 * (mu_grid[1:] + mu_grid[:-1]),
        [mu_grid[-1]]
    ))
    plt.ion()
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 8), sharex=True, sharey=True)
    mappable = ax[0].pcolormesh(mu_edges, wave_edges, lw_I, norm=LogNorm())
    fig.colorbar(mappable, ax=ax[0])
    ax[0].set_title("Plane-Parallel")
    mappable = ax[1].pcolormesh(mu_edges, wave_edges, spherical_I, norm=LogNorm())
    fig.colorbar(mappable, ax=ax[1])
    ax[1].set_title("Depth Rays")
    fig.savefig("FSComparison.png", dpi=300)'''

#--------------------------------------------------------------------------------------------------------------

    #print (t1,t2)

    # now we should figure out if this is a 'core' ray or an 'outer' ray:
    # Maybe this is not needed:

    # that is decided by the dy:
    #core_ray = True

    #R_core = const.R_sun.value
    #print (R_core)
    #dz_max = 2E6 # in m, hard-coded, FIX!
    #R_max = const.R_sun.value+dz_max
    
    # Here remember that R = 1.0 corresponds to the R_max
    #if (dy > R_core/R_max):
    #    core_ray = False

    #print("info:: I am a core ray: ", core_ray)

    #sample_ts = 0
    #if (core_ray == False):
    #    sample_ts = np.linspace(t1, t2, num_sample_points)
    #else:
    #    t1 = 