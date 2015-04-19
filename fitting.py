# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:49:13 2015

@author: ttshimiz
"""

import numpy as np
import emcee
import astropy.constants as c
import astropy.units as u
from glob import glob

# CONSTANTS
c_micron = c.c.to(u.micron/u.s).value


def log_like(params, x, y, yerr, sed_model, fixed, filts, filt_all):

    model_fluxes = calc_model(x, params, sed_model, fixed, filts, filt_all)

    return -0.5*(np.sum((y-model_fluxes)**2/yerr**2+np.log(2*np.pi*yerr**2)))


def log_prior(params, sed_model, fixed):

    bounds = np.array([sed_model.bounds[n] for n in sed_model.param_names])
    bounds = bounds[~fixed]
    lp = map(uniform_prior, params, bounds)

    return sum(lp)


def log_post(params, x, y, yerr, sed_model, fixed, filts, filt_all):

    lprior = log_prior(params, sed_model, fixed)
    if not np.isfinite(lprior):
        return -np.inf
    else:
        llike = log_like(params, x, y, yerr, sed_model, fixed, filts, filt_all)
        if not np.isfinite(llike):
            return -np.inf
        else:
            return lprior + llike


def uniform_prior(x, bounds):
    if bounds[0] is None:
        bounds[0] = -np.inf
    if bounds[1] is None:
        bounds[1] = np.inf

    if (x >= bounds[0]) & (x <= bounds[1]):
        return 0
    else:
        return -np.inf


# Calculate monochromatic flux densities of the model using the
# filter transmission curves and redshift of the source
def calc_model(waves, params, sed_model, fixed, filts, filt_all):
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

    # Dummy model
    dummy = sed_model.copy()
    dummy.parameters[~fixed] = params
    fwaves = filt_all.filter_waves
    zz = dummy.redshift
    zcorr = 1 + zz

    func = filt_all.calc_mono_flux

    model_fluxes = np.array([func(f, fwaves[f],
                                  dummy(fwaves[f]/zcorr)) for f in filts])

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

    def fit(self, model, x, y, yerr=None, filts=None,
            best=np.median, errs=(16, 84)):

        mod = model.copy()
        fixed = np.array([mod.fixed[n] for n in mod.param_names])
        self.ndims = np.sum(~fixed)

        # Use the current model parameters as the initial values
        init = mod.parameters[~fixed]
        init_walkers = [init + 1e-4*np.random.randn(self.ndims)
                        for k in range(self.nwalkers)]

        # Create the Filter object that stores all of the filters
        filt_all = Filters()

        # Setup the MCMC sampler
        mcmc = emcee.EnsembleSampler(self.nwalkers, self.ndims, log_post,
                                     args=(x, y, yerr, mod, fixed,
                                           filts, filt_all),
                                     threads=self.threads)

        mcmc.run_mcmc(init_walkers, self.nsteps)

        mod.chain = mcmc.chain[:, :, :].reshape(-1, self.ndims)
        mod.chain_nb = mcmc.chain[:, self.nburn:, :].reshape(-1, self.ndims)

        if self.threads > 1:
            mcmc.pool.close()

        mod.parameters[~fixed] = best(mod.chain_nb, axis=0)
        mod.param_errs = np.zeros((len(mod.parameters), 2))
        mod.param_errs[~fixed] = np.percentile(mod.chain_nb, q=errs, axis=0).T

        return mod


class Filters(object):

    def __init__(self):

        fn = glob('/Users/ttshimiz/Github/bat-agn-sed-fitting/Filters/*.txt')
        self.names = [x.split('/')[-1].split('.')[0] for x in fn]
        self.filter_trans = {}
        self.filter_waves = {}

        for i, n in enumerate(fn):
            data = np.loadtxt(n)
            fw = data[:, 0]
            ft = data[:, 1]
            ind = ft > 3e-3
            fw = fw[ind]
            ft = ft[ind]
            isort = np.argsort(fw)
            fw = fw[isort]
            ft = ft[isort]
            self.filter_trans[self.names[i]] = ft
            self.filter_waves[self.names[i]] = fw

    def get_waves(self, f):
        return self.filter_waves[f]

    def get_trans(self, f):
        return self.filter_trans[f]

    def calc_mono_flux(self, filt, sed_waves, sed):
        ft = self.filter_trans[filt]
        fw = self.filter_waves[filt]

        interp_sed = np.interp(fw, sed_waves, sed, left=0, right=0)

        integ_num = np.trapz(ft*interp_sed, x=c_micron/fw)
        integ_denom = np.trapz(ft, x=c_micron/fw)

        return integ_num/integ_denom
