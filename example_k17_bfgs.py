"""
Example ThirdWay experimentations.
"""
from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from thirdway.lightcurve import LightCurve, generate_lc_depth, kepler17_params_db
from thirdway.fitting import peak_finder, summed_gaussians, gaussian
from astropy.utils.console import ProgressBar

# Load light curve from jrad's text archive
light_curve_path = 'data/kepler17_whole.dat'
BJDREF = 2454833.
depth = 0.13413993**2
jd_minus_bjdref, flux, error = np.loadtxt(light_curve_path, unpack=True)
jd = jd_minus_bjdref + BJDREF
kepler17_params = kepler17_params_db()

# construct light curve object from those data
whole_lc = LightCurve(times=jd, fluxes=flux, errors=error)
transits = LightCurve(**whole_lc.mask_out_of_transit(kepler17_params)
                      ).get_transit_light_curves(kepler17_params)

# The short cadence data begin after the 137th transit, so ignore all transits before then:
transits = transits[137:]

plots = True

delta_chi2 = {}#np.zeros(len(transits))

with ProgressBar(len(transits)) as bar:
    for i, lc in enumerate(transits):
        #lc.plot()
        # Remove linear out-of-transit trend from transit
        lc.remove_linear_baseline(kepler17_params)
        residuals = lc.fluxes - generate_lc_depth(lc.times_jd, depth, kepler17_params)

        best_fit_params = peak_finder(lc.times.jd, residuals, lc.errors,
                                      kepler17_params)
        
        transit_model = generate_lc_depth(lc.times_jd, depth, kepler17_params)
        chi2_transit = np.sum((lc.fluxes - transit_model)**2/lc.errors**2)/len(lc.fluxes)

        gaussian_model = summed_gaussians(lc.times.jd, best_fit_params)
        
        if best_fit_params is not None:
            split_input_parameters = np.split(np.array(best_fit_params), len(best_fit_params)/3)
            delta_chi2[i] = []
            for amplitude, t0, sigma in split_input_parameters:
                model_i = gaussian(lc.times.jd, amplitude, t0, sigma)
                chi2_bumps = np.sum((lc.fluxes - transit_model - model_i)**2/lc.errors**2)/len(lc.fluxes)
                delta_chi2[i].append(np.abs(chi2_transit - chi2_bumps))
        
            if plots:
                fig, ax = plt.subplots(3, 1, figsize=(8, 14), sharex=True)

                ax[0].errorbar(lc.times.jd, lc.fluxes, lc.errors, fmt='.', color='k')
                ax[0].plot(lc.times.jd, transit_model, 'r')
                ax[0].set(ylabel='Flux')

                ax[1].axhline(0, color='gray', ls='--')
                ax[1].errorbar(lc.times.jd, lc.fluxes - transit_model, fmt='.', color='k')
                ax[1].plot(lc.times.jd, gaussian_model, color='r')
                ax[1].set_ylabel('Transit Residuals')

                ax[2].axhline(0, color='gray', ls='--')
                ax[2].errorbar(lc.times.jd, lc.fluxes - transit_model - gaussian_model, fmt='.', color='k')
                ax[2].set_ylabel('Gaussian Residuals')
                ax[2].set_title(r'$Delta \chi^2$ = '+'{0}'.format(delta_chi2[i]))

                fig.tight_layout()
                fig.savefig('plots/{0:03d}.png'.format(i), bbox_inches='tight')
                #plt.show()
                plt.close()
        
        bar.update()

print(delta_chi2.values())
all_delta_chi2 = np.concatenate(delta_chi2.values()).ravel()

fig, ax = plt.subplots(1,figsize=(12, 6))
ax.plot(np.log10(all_delta_chi2), '.')
plt.show()
