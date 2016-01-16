
from __future__ import absolute_import, print_function
import numpy as np
import emcee
from scipy import optimize, signal
import matplotlib.pyplot as plt


def gaussian(times, amplitude, t0, sigma):
    return amplitude * np.exp(-0.5*(times - t0)**2/sigma**2)


def summed_gaussians(times, input_parameters):
    """
    Take a list of gaussian input parameters (3 parameters per gaussian), make a model of the
    sum of all of those gaussians.
    """
    model = np.zeros(len(times), dtype=np.float64)
    if input_parameters is not None:
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


def get_in_transit_bounds(x, params, duration_fraction=0.7):
    phased = (x - params.t0) % params.per
    near_transit = ((phased < params.duration*(0.5*duration_fraction)) |
                    (phased > params.per - params.duration*(0.5*duration_fraction)))
    if np.count_nonzero(near_transit) == 0:
        near_transit = 0
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

#def fmin_fitter(times, residuals, errors):
#    n_peaks = 4
#    peak_times = np.linspace(times.min(), times.max(), n_peaks+2)[1:-1]
#    peak_amplitudes = n_peaks*[2*np.std(residuals)]
#    peak_sigmas = n_peaks*[5./60/24] # 5 min
#    input_parameters = np.vstack([peak_amplitudes, peak_times, peak_sigmas]).T.ravel()
#    
#    min_t, max_t = get_in_transit_bounds(times)
#    times_bounds = n_peaks*[(min_t, max_t)]
#    amplitudes_bounds = n_peaks*[(0, 1)]
#    sigmas_bounds = n_peaks*[(1./60/24, max_t - min_t)]
#    #bounds = np.vstack([amplitudes_bounds, times_bounds, sigmas_bounds]).T.ravel()
#    bounds = []
#    for i, j, k in zip(amplitudes_bounds, times_bounds, sigmas_bounds):
#        bounds.extend([i, j, k])
#    
#    print('bounds shape: {0}'.format(np.shape(bounds)), '\ninputs: {0}'.format(input_parameters.shape))
#    def chi2(*args, **kwargs):
#        return -2*lnlike(*args, **kwargs)
#    results = optimize.fmin_l_bfgs_b(chi2, input_parameters, bounds=bounds, 
#                                     args=(times, residuals, errors), approx_grad=True)
#    return results


def lnlike(theta, x, y, yerr):
    model = summed_gaussians(x, theta)
    return -0.5*np.sum((y-model)**2/yerr**2)


def lnprob(theta, x, y, yerr, lower_t_bound, upper_t_bound):
    lp = lnprior(theta, y, lower_t_bound, upper_t_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def run_emcee(times, residuals, errors, params, n_peaks=4, burnin=0.7):

    # Create n_peaks evenly spaced peaks throught residuals:

    lower_t_bound, upper_t_bound = get_in_transit_bounds(times, params)

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
    n_steps = 25000
    sampler.run_mcmc(pos, n_steps)
    burnin_len = int(burnin*n_steps)
    samples = sampler.chain[:, burnin_len:, :].reshape((-1, ndim))
    return sampler, samples

       
def chi2(theta, x, y, yerr):
    model = summed_gaussians(x, theta)
    return np.sum((y-model)**2/yerr**2)


def peak_finder(times, residuals, errors, params, n_peaks=4, plots=False, verbose=False):
    # http://stackoverflow.com/a/25666951
    # Convolve residuals with a gaussian, find relative maxima
    n_points_kernel = 100
    window = signal.general_gaussian(n_points_kernel+1, p=1, sig=12)
    filtered = signal.fftconvolve(window, residuals)
    filtered = (np.average(residuals) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, int(-n_points_kernel/2))
    maxes = signal.argrelmax(filtered)[0]
    maxes = maxes[maxes < len(residuals)]
    maxes = maxes[residuals[maxes] > 0]

    lower_t_bound, upper_t_bound = get_in_transit_bounds(times, params)
    maxes_in_transit = maxes[(times[maxes] < upper_t_bound) & 
                             (times[maxes] > lower_t_bound)]

    if len(maxes_in_transit) == 0:
        if verbose: 
            print('no maxes found')
        return None

    peak_times = times[maxes_in_transit]
    peak_amplitudes = residuals[maxes_in_transit]#len(peak_times)*[3*np.std(residuals)]
    peak_sigmas = len(peak_times)*[4./60/24] # 5 min
    input_parameters = np.vstack([peak_amplitudes, peak_times, peak_sigmas]).T.ravel()
    # result = optimize.fmin(chi2, input_parameters, args=(times, residuals, errors))
    result = optimize.fmin_bfgs(chi2, input_parameters, disp=False,
                                args=(times, residuals, errors))        
    #print(result, result == input_parameters)
    
    if plots:
        fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
        ax[0].errorbar(times, residuals, fmt='.', color='k')
        [ax[0].axvline(t) for t in times[maxes_in_transit]]
        ax[0].plot(times, summed_gaussians(times, input_parameters), 'r')
        #ax[1].errorbar(times, gaussian_model, fmt='.', color='r')
        ax[0].axhline(0, color='k', ls='--')
        ax[0].set_ylabel('Transit Residuals')
    
        ax[1].errorbar(times, residuals, fmt='.', color='k')
        ax[1].plot(times, summed_gaussians(times, result), 'r')
        #ax[1].errorbar(times, gaussian_model, fmt='.', color='r')
        ax[1].axhline(0, color='k', ls='--')
        ax[1].set_ylabel('Residuals')
    
        ax[2].errorbar(times, residuals - summed_gaussians(times, result), fmt='.', color='k')
        #ax[1].errorbar(times, gaussian_model, fmt='.', color='r')
        ax[2].axhline(0, color='k', ls='--')
        ax[2].set_ylabel('Residuals')

        fig.tight_layout()
        plt.show()
    return result
