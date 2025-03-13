import numpy as np
import matplotlib.pyplot as plt
import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from matplotlib.colors import LogNorm

def compute_intersections(p0, d):
    # NOTE(cmo): Units of Rs
    a = 1.0
    b = 2.0 * (p0 @ d)
    c = p0 @ p0 - 1.0
    delta = b*b - 4.0 * a * c

    if delta < 0:
        return None
    t1 = (-b - np.sqrt(delta)) / (2.0 * a)
    t2 = (-b + np.sqrt(delta)) / (2.0 * a)
    return t1, t2

def sphere_trace_ctx(ctx, mu, num_sample_points=20_000):
    p0 = np.array([0.0, 0.0, 1.0])
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

mu_grid = np.linspace(0.01, 0.1, 19)

if __name__ == "__main__":
    falc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"], Nthreads=12)
    falc_ctx.depthData.fill = True
    falc_ctx.formal_sol_gamma_matrices()

    plane_parallel_tabulate = pw.tabulate_bc(falc_ctx, mu_grid=mu_grid)
    lw_I = plane_parallel_tabulate["I"]

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
    fig.savefig("FSComparison.png", dpi=300)


