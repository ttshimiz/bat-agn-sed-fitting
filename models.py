# -*- coding: utf-8 -*-
"""
Module to hold all of the models that will be used to fit the BAT AGN SED's.
Models currently include:
    1. Single Temperature Greybody
    2. Two Temperature Greybody
    3. Casey 2012 Greybody+Powerlaw
    4. DecompIR templates
    5. Dale et al 2014 models

Models will all be subclasses of the Astropy Fittable1DModel class.

Created on Thu Apr  9 16:25:56 2015

@author: ttshimiz
"""

from astropy.modeling import Fittable1DModel, Parameter
from astropy.analytic_functions import blackbody_nu
import astropy.units as u
import astropy.constants as c
import numpy as np


# Single Temperature Greybody
class Greybody(Fittable1DModel):
    """
    Single temperature greybody.
    Parameters include dust mass, emissivity, and dust temperature

    Equation:
    S_grey = mdust*kappa0*(nu/(c/lamdda_norm))^beta*B(nu,tdust)/lumD^2

    Parameters:
        mdust: dust mass
        kappa0: normalization of the absorption coefficient
                Default is 0.192 m^2/kg
        lambda_norm: wavelength at which the absorption coefficient
                     is normalized
                     Default is 350 micron.
        beta: dust emissivity
              Default is 2.0.
        tdust: dust temperature
        lumD: luminosity distance
        B(nu,tdust): is the standard blackbody equation
    """

    mdust = Parameter()
    tdust = Parameter()
    beta = Parameter(default=2.0)

    def __init__(self, mdust, tdust, beta, lumD=1.0, kappa0=0.192,
                 lambda_norm=350.):

        self.set_kappa0(kappa0)
        self.set_lambda_norm(lambda_norm)
        self.set_lumD(lumD)

        super(Greybody, self).__init__(mdust, tdust, beta)

    def evaluate(self, x, mdust, tdust, beta):
        md = (10**mdust*u.Msun).to(u.g).value
        n = c.c.to(u.micron/u.s).value/x
        return (md * self.k0_cgs * (n / self.nu_norm)**beta * np.pi *
                blackbody_nu(n, tdust).value / self.lumD_cm**2)*1e23

    def set_kappa0(self, k0):
        self.kappa0 = k0*u.m**2/u.kg
        self.k0_cgs = (k0*u.m**2/u.kg).cgs.value   # Convert to cm**2/g

    def set_lambda_norm(self, lnorm):
        self.lambda_norm = lnorm*u.micron
        self.nu_norm = (c.c.to(u.micron/u.s)/lnorm).value  # Convert to Hz

    def set_lumD(self, ld):
        self.lumD = ld*u.Mpc
        self.lumD_cm = (ld*u.Mpc).cgs.value  # Convert to cm


class TwoTempGreybody(Fittable1DModel):

    mdust_warm = Parameter()
    tdust_warm = Parameter(bounds=(35, 80))
    beta_warm = Parameter(default=2.0, bounds=(0, 5))
    mdust_cold = Parameter()
    tdust_cold = Parameter(bounds=(0, 35))
    beta_cold = Parameter(default=2.0, bounds=(0, 5))

    def __init__(self, mdust_warm, tdust_warm, beta_warm,
                 mdust_cold, tdust_cold, beta_cold,
                 lumD=1.0, kappa0=0.192, lambda_norm=350.):

        self.set_kappa0(kappa0)
        self.set_lambda_norm(lambda_norm)
        self.set_lumD(lumD)

        super(TwoTempGreybody, self).__init__(mdust_warm,
                                              tdust_warm,
                                              beta_warm,
                                              mdust_cold,
                                              tdust_cold,
                                              beta_cold)

    def evaluate(self, x, mdust_warm, tdust_warm, beta_warm,
                 mdust_cold, tdust_cold, beta_cold):

        mwarm = (10**mdust_warm*u.Msun).to(u.g).value
        mcold = (10**mdust_cold*u.Msun).to(u.g).value
        n = c.c.to(u.micron/u.s).value/x
        flux_warm = (mwarm * self.k0_cgs *
                     (n / self.nu_norm)**beta_warm *
                     np.pi * blackbody_nu(n, tdust_warm).value /
                     self.lumD_cm**2)*1e23

        flux_cold = (mcold * self.k0_cgs *
                     (n / self.nu_norm)**beta_cold *
                     np.pi * blackbody_nu(n, tdust_cold).value /
                     self.lumD_cm**2)*1e23
        return flux_warm + flux_cold

    def set_kappa0(self, k0):
        self.kappa0 = k0*u.m**2/u.kg
        self.k0_cgs = (k0*u.m**2/u.kg).cgs.value   # Convert to cm**2/g

    def set_lambda_norm(self, lnorm):
        self.lambda_norm = lnorm*u.micron
        self.nu_norm = (c.c.to(u.micron/u.s)/lnorm).value  # Convert to Hz

    def set_lumD(self, ld):
        self.lumD = ld*u.Mpc
        self.lumD_cm = (ld*u.Mpc).cgs.value  # Convert to cm


class GreybodyPowerlaw(Fittable1DModel):

    mdust = Parameter()
    tdust = Parameter()
    beta = Parameter(default=2.0, bounds=(0, 5))
    pownorm = Parameter()
    alpha = Parameter(default=2.0, bounds=(-5, 5))
    wturn = Parameter(default=40.0, bounds=(20, 100))

    def __init__(self, mdust, tdust, beta, pownorm, alpha, wturn,
                 lumD=1.0, kappa0=0.192, lambda_norm=350.):

        self.set_kappa0(kappa0)
        self.set_lambda_norm(lambda_norm)
        self.set_lumD(lumD)

        super(GreybodyPowerlaw, self).__init__(mdust, tdust, beta,
                                               pownorm, alpha, wturn)

    def evaluate(self, x, mdust, tdust, beta,
                 pownorm, alpha, wturn):

        md = (10**mdust*u.Msun).to(u.g).value
        n = c.c.to(u.micron/u.s).value/x
        flux_grey = (md * self.k0_cgs *
                     (n / self.nu_norm)**beta *
                     np.pi * blackbody_nu(n, tdust).value /
                     self.lumD_cm**2)*1e23

        flux_plaw = (10**pownorm) * (x/wturn)**alpha * np.exp(-(x/wturn)**2)

        return flux_grey + flux_plaw

    def eval_grey(self, x):
        md = (10**self.mdust*u.Msun).to(u.g).value
        n = c.c.to(u.micron/u.s).value/x
        flux_grey = (md * self.k0_cgs *
                     (n / self.nu_norm)**self.beta *
                     np.pi * blackbody_nu(n, self.tdust).value /
                     self.lumD_cm**2)*1e23
        return flux_grey

    def eval_plaw(self, x):
        return ((10**self.pownorm) * (x/self.wturn)**self.alpha *
                np.exp(-(x/self.wturn)**2))

    def set_kappa0(self, k0):
        self.kappa0 = k0*u.m**2/u.kg
        self.k0_cgs = (k0*u.m**2/u.kg).cgs.value   # Convert to cm**2/g

    def set_lambda_norm(self, lnorm):
        self.lambda_norm = lnorm*u.micron
        self.nu_norm = (c.c.to(u.micron/u.s)/lnorm).value  # Convert to Hz

    def set_lumD(self, ld):
        self.lumD = ld*u.Mpc
        self.lumD_cm = (ld*u.Mpc).cgs.value  # Convert to cm
