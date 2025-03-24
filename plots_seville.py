import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt 
import astropy.constants as const

import matplotlib
font = {'size'   : 18}

matplotlib.rc('font', **font)

spec_synth = fits.open("demonstration.fits")[0].data
limbdistances = fits.open("demonstration.fits")[1].data
paths = fits.open("demonstration.fits")[2].data
taus = fits.open("demonstration.fits")[3].data


Rs = const.R_sun.value

mu = np.sqrt(1.0 - (Rs-limbdistances)**2.0/Rs**2.0)

plt.figure(figsize=[9,5])
plt.plot(limbdistances/1E3, paths/1e3, label='spherical', linewidth=2)
plt.plot(limbdistances/1E3, paths[-1]/1e3*mu[-1]/mu, label='plane-parallel', linewidth=2)
plt.legend()
plt.xlim([0,20000])
plt.xlabel("Limb distance [km]")
plt.ylabel("Path length [km]")
plt.tight_layout()
plt.savefig("paths.png",bbox_inches='tight')

plt.figure(figsize=[9,5])
plt.plot(mu, paths/1e3, label='spherical', linewidth=2)
plt.plot(mu, paths[-1]/1e3*mu[-1]/mu, label='plane-parallel', linewidth=2)
plt.legend()
#plt.xlim([0,20000])
plt.xlabel("$\mu$")
plt.ylabel("Path length [km]")
plt.tight_layout()
plt.savefig("pathsmu.png",bbox_inches='tight')

plt.figure(figsize=[9,5])
plt.plot(limbdistances/1E3, taus[:,650], label='spherical', linewidth=2)
plt.plot(limbdistances/1E3, taus[-1,650]*mu[-1]/mu, label='plane-parallel', linewidth=2)
plt.legend()
plt.xlim([0,20000])
plt.xlabel("Limb distance [km]")
plt.ylabel("Optical depth")
plt.tight_layout()
plt.savefig("taus.png",bbox_inches='tight')

plt.figure(figsize=[9,5])
plt.plot(mu, taus[:,650], label='spherical', linewidth=2)
plt.plot(mu, taus[-1,650]*mu[-1]/mu, label='plane-parallel', linewidth=2)
plt.legend()
#plt.xlim([0,20000])
plt.xlabel("$\mu$")
plt.ylabel("Optical depth")
plt.tight_layout()
plt.savefig("tausmu.png",bbox_inches='tight')

plt.figure(figsize=[9,5])
plt.semilogy(limbdistances/1E3, taus[:,650], label='spherical', linewidth=2)
plt.semilogy(limbdistances/1E3, taus[-1,650]*mu[-1]/mu, label='plane-parallel', linewidth=2)
plt.legend()
plt.xlim([0,20000])
plt.xlabel("Limb distance [km]")
plt.ylabel("Optical depth")
plt.tight_layout()
plt.savefig("tauslog.png",bbox_inches='tight')

plt.figure(figsize=[9,5])
plt.semilogy(mu, taus[:,650], label='spherical', linewidth=2)
plt.semilogy(mu, taus[-1,650]*mu[-1]/mu, label='plane-parallel', linewidth=2)
plt.legend()
#plt.xlim([0,20000])
plt.xlabel("$\mu$")
plt.ylabel("Optical depth")
plt.tight_layout()
plt.savefig("tauslogmu.png",bbox_inches='tight')