# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:49:13 2015

@author: ttshimiz
"""

import numpy as np
import emcee
import astropy.constants as c
import astropy.units as u

# FILTERS
w1_filter = np.loadtxt('Filters/WISE/RSR-W1.EE.txt')
w2_filter = np.loadtxt('Filters/WISE/RSR-W2.EE.txt')
w3_filter = np.loadtxt('Filters/WISE/RSR-W3.EE.txt')
w4_filter = np.loadtxt('Filters/WISE/RSR-W4.EE.txt')
h70_filter = np.loadtxt('Filters/Herschel/PACS70.txt')
h70_ind = (h70_filter[:, 0] > 50) & (h70_filter[:, 0] < 105)
h70_filter = h70_filter[h70_ind]
h160_filter = np.loadtxt('Filters/Herschel/PACS160.txt')
h250_filter = np.loadtxt('Filters/Herschel/PSW.txt')
h250_ind = (h250_filter[:, 0] > 180) & (h250_filter[:, 0] < 310)
h250_filter = h250_filter[h250_ind]
h250_filter = h250_filter[-1::-1]
h350_filter = np.loadtxt('Filters/Herschel/PMW.txt')
h350_ind = (h350_filter[:, 0] > 250) & (h350_filter[:, 0] < 450)
h350_filter = h350_filter[h350_ind]
h350_filter = h350_filter[-1::-1]
h500_filter = np.loadtxt('Filters/Herschel/PLW.txt')
h500_ind = (h500_filter[:, 0] > 360) & (h500_filter[:, 0] < 620)
h500_filter = h500_filter[h500_ind]
h500_filter = h500_filter[-1::-1]

# CONSTANTS
c_micron = c.c.to(u.micron/u.s).value


def log_like(params, x, y, yerr, sed_model, fixed):

    model_fluxes = calc_model(x, params, sed_model, fixed)

    return -0.5*(np.sum((y-model_fluxes)**2/yerr**2+np.log(2*np.pi*yerr**2)))


def log_prior(params, sed_model, fixed):

    lp = 0

    for i, n in enumerate(np.array(sed_model.param_names)[~fixed]):
        low, up = sed_model.bounds[n]
        if up is None:
            up = np.inf
        if low is None:
            low = -np.inf

        if (params[i] < up) & (params[i] > low):
            lp += 0
        else:
            lp = -np.inf
    return lp


def log_post(params, x, y, yerr, sed_model, fixed):

    llike = log_like(params, x, y, yerr, sed_model, fixed)
    lprior = log_prior(params, sed_model, fixed)

    if not np.isfinite(lprior) or not np.isfinite(llike):
        return -np.inf
    return lprior + llike


def convolve_with_trans(sed_waves, sed_flux, filt_waves, filt_trans):
    """
    Function to integrate the SED over a bandpass with
    a defined transmission curve

    Input Parameters
    ----------------
        sed_waves   = wavelengths at which the SED is defined
        sed_flux    = flux densities of the SED
        filt_waves  = wavelengths at which the filter is defined
        filt_trans  = transmission of the filter

    Output
    ------
        fnu = integrated flux density

    """

    # Interpolate the transmission to the same wavelengths as the SED
    # Use a value of 0 for where the filter isn't defined
    if ~all(sed_waves == filt_waves):
        interp_trans = np.interp(sed_waves, filt_waves, filt_trans,
                                 left=0, right=0)
    else:
        interp_trans = filt_trans

    # Integrate over all wavelengths (but use the frequencies
    # since the SED is in Jy)
    integ_num = np.trapz(c_micron/sed_waves, interp_trans*sed_flux)
    integ_denom = np.trapz(c_micron/sed_waves, interp_trans)

    return integ_num/integ_denom


# Calculate monochromatic flux densities of the model using the
# filter transmission curves and redshift of the source
def calc_model(waves, params, sed_model, fixed):
    """
    For each observed wavelength we need to determine which filter to use.
    3.4, 4.6, 12., and 22. micron will be assumed to be WISE filters.
    70 - 500 micron will be assumed to be Herschel.
    In the future, I need to add in more possibilities.
    Then the wavelengths in the filter need to be redshifted to get the
    rest frame wavelengths.
    The model will be calculated at these rest frame wavelengths,
    then redshifted back to observed frame where it will be convolved
    with the transmission curve.
    """

    # Array to store the calculated model fluxes
    model_fluxes = np.zeros(np.shape(waves))

    # Dummy model
    dummy = sed_model.copy()

    for i, w in enumerate(waves):
        if w == 3.4:
            filter = w1_filter
        elif w == 4.6:
            filter = w2_filter
        elif w == 12.:
            filter = w3_filter
        elif w == 22.:
            filter = w4_filter
        elif w == 70.:
            filter = h70_filter
        elif w == 160.:
            filter = h160_filter
        elif w == 250.:
            filter = h250_filter
        elif w == 350.:
            filter = h350_filter
        elif w == 500.:
            filter = h500_filter

        filter_waves = filter[:, 0]
        filter_trans = filter[:, 1]

        # Transform to rest frame
        rest_waves = filter_waves / (1 + sed_model.redshift)

        # Calculate the model at the rest frame wavelengths
        dummy.parameters[~fixed] = params
        rest_total_model = dummy(rest_waves)

        # Convolve the SED with the transmission curve
        mono_flux = convolve_with_trans(filter_waves, rest_total_model,
                                        filter_waves, filter_trans)

        model_fluxes[i] = mono_flux

    return model_fluxes


class SEDBayesFitter(object):

    def __init__(self, nwalkers=50, nsteps=1000, nburn=200, threads=4):

        self.set_nwalkers(nwalkers)
        self.set_nsteps(nsteps)
        self.set_nburn(nburn)
        self.set_nthreads(threads)

    def set_nwalkers(self, nw):
        self.nwalkers = nw

    def set_nsteps(self, ns):
        self.nsteps = ns

    def set_nburn(self, nb):
        self.nburn = nb

    def set_nthreads(self, nt):
        self.threads = nt

    def fit(self, model, x, y, yerr):

        fixed = np.array([model.fixed.values()])
        self.ndims = np.sum(~fixed)

        # Use the current model parameters as the initial values
        init = model.parameters[~fixed]
        init_walkers = [init + 1e-4*np.random.randn(self.ndims)
                        for k in range(self.nwalkers)]

        # Setup the MCMC sampler
        mcmc = emcee.EnsembleSampler(self.nwalkers, self.ndims, log_post,
                                     args=(x, y, yerr, model, fixed),
                                     threads=self.threads)

        mcmc.run_mcmc(init_walkers, self.nsteps)

        self.chain = mcmc.chain[:, :, :].reshape(-1, self.ndims)
        self.chain_nb = mcmc.chain[:, self.nburn:, :].reshape(-1, self.ndims)

        mcmc.pool.close()

        model.parameters[~fixed] = np.median(self.chain_nb, axis=0)
