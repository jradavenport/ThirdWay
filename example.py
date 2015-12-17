"""
Example ThirdWay experimentations.
"""
from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from thirdway.lightcurve import LightCurve, generate_lc_depth
from thirdway.fitting import run_emcee, summed_gaussians

# Load light curve from jrad's text archive
light_curve_path = 'data/kepler17_whole.dat'
BJDREF = 2454833.
jd_minus_bjdref, flux, error = np.loadtxt(light_curve_path, unpack=True)
jd = jd_minus_bjdref + BJDREF

# construct light curve object from those data
whole_lc = LightCurve(times=jd, fluxes=flux, errors=error)
transits = LightCurve(**whole_lc.mask_out_of_transit()).get_transit_light_curves()

# The short cadence data begin after the 137th transit, so ignore all transits before then:
transits = transits[137:]


lc = transits[104]
# Remove linear out-of-transit trend from transit
lc.remove_linear_baseline()
depth = 0.13413993**2
residuals = lc.fluxes - generate_lc_depth(lc.times_jd, depth)

# Fit the transit model residuals for n_peaks gaussians
sampler, samples = run_emcee(lc.times.jd, residuals, lc.errors, n_peaks=4)

# Plot the results

import corner
corner.corner(samples)
median_params = np.median(samples, axis=0)
transit_model = generate_lc_depth(lc.times_jd, depth)
gaussian_model = summed_gaussians(lc.times.jd, median_params)

fig, ax = plt.subplots(3, 1, figsize=(8, 14), sharex=True)

ax[0].errorbar(lc.times.jd, lc.fluxes, lc.errors, fmt='.', color='k')
ax[0].plot(lc.times.jd, transit_model, 'r')
ax[0].set(ylabel='Flux')

ax[1].errorbar(lc.times.jd, lc.fluxes - transit_model, fmt='.', color='k')
ax[1].errorbar(lc.times.jd, gaussian_model, fmt='.', color='r')
ax[1].axhline(0, color='r')
ax[1].set_ylabel('Transit Residuals')

ax[2].errorbar(lc.times.jd, lc.fluxes - transit_model - gaussian_model, fmt='.', color='k')
ax[2].axhline(0, color='r')
ax[2].set_ylabel('Gaussian Residuals')

fig.tight_layout()
plt.show()
#plt.show()