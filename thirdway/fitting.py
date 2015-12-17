
from __future__ import absolute_import, print_function
import numpy as np
import emcee
from .lightcurve import params


def gaussian(times, amplitude, t0, sigma):
    return amplitude * np.exp(-0.5*(times - t0)**2/sigma**2)


def summed_gaussians(times, input_parameters):
    """
    Take a list of gaussian input parameters (3 parameters per gaussian), make a model of the
    sum of all of those gaussians.
    """
    model = np.zeros(len(times), dtype=np.float64)
    split_input_parameters = np.split(np.array(input_parameters), len(input_parameters)/3)
    for amplitude, t0, sigma in split_input_parameters:
        model += gaussian(times, amplitude, t0, sigma)

    return model

# def leastsq_fitter(times, residuals, errors, n_peaks=4):
#
#     # Create n_peaks evenly spaced peaks throught residuals:
#
#     peak_times = np.linspace(times.min(), times.max(), n_peaks+2)[1:-1]
#     peak_amplitudes = n_peaks*[4*np.std(residuals)]
#     peak_sigmas = n_peaks*[5./60/24] # 5 min
#
#     input_parameters = np.vstack([peak_amplitudes, peak_times, peak_sigmas]).T.ravel()
#
#     minimize_this = lambda p: np.sum((residuals - summed_gaussians(times, p))**2/errors**2)
#
#     #p, success = optimize.leastsq(minimize_this, input_parameters)
#     #p = optimize.minimize(minimize_this, input_parameters)
#     p = optimize.fmin(minimize_this, input_parameters, disp=False)
#     return p


def get_in_transit_bounds(x, params=params, duration_fraction=0.7):
    phased = (x - params.t0) % params.per
    near_transit = ((phased < params.duration*(0.5*duration_fraction)) |
                    (phased > params.per - params.duration*(0.5*duration_fraction)))
    return (x[near_transit].min(), x[near_transit].max())


def lnprior(theta, y, lower_t_bound, upper_t_bound):
    amplitudes = theta[::3]
    t0s = theta[1::3]
    sigmas = theta[2::3]

    if (((0 <= amplitudes) & (amplitudes < 200*np.std(y))).all() and
        ((lower_t_bound < t0s) & (t0s < upper_t_bound)).all() and
        ((1./60/24 < sigmas) & (sigmas < upper_t_bound - lower_t_bound)).all()):
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr):
    model = summed_gaussians(x, theta)
    return -0.5*np.sum((y-model)**2/yerr**2)


def lnprob(theta, x, y, yerr, lower_t_bound, upper_t_bound):
    lp = lnprior(theta, y, lower_t_bound, upper_t_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def run_emcee(times, residuals, errors, n_peaks=4, burnin=0.7):

    # Create n_peaks evenly spaced peaks throught residuals:

    lower_t_bound, upper_t_bound = get_in_transit_bounds(times)

    peak_times = np.linspace(lower_t_bound, upper_t_bound, n_peaks+2)[1:-1]
    peak_amplitudes = n_peaks*[4*np.std(residuals)]
    peak_sigmas = n_peaks*[4./60/24] # 5 min

    input_parameters = np.vstack([peak_amplitudes, peak_times, peak_sigmas]).T.ravel()

    ndim, nwalkers = len(input_parameters), 10*len(input_parameters)
    #pos = [input_parameters + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    pos = []

    while len(pos) < nwalkers:
        realization = input_parameters + 1e-3*np.random.randn(ndim)
        if lnprior(realization, residuals, lower_t_bound, upper_t_bound) == 0.0:
            pos.append(realization)

    print('starting positions')
    pool = emcee.interruptible_pool.InterruptiblePool(processes=4)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(times, residuals, errors, lower_t_bound, upper_t_bound),
                                    pool=pool)
    n_steps = 20000
    sampler.run_mcmc(pos, n_steps)
    burnin_len = int(burnin*n_steps)
    samples = sampler.chain[:, burnin_len:, :].reshape((-1, ndim))
    return sampler, samples
