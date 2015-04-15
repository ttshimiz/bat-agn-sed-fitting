# -*- coding: utf-8 -*-
"""
Module to hold all of the models that will be used to fit the BAT AGN SED's.
Models currently include:
    1. Single Temperature Greybody
    2. Two Temperature Greybody
    3. Casey 2012 Greybody+Powerlaw
    4. DecompIR templates
    5. Dale et al 2014 models

Most models will be subclasses of the Astropy Fittable1DModel class.

Created on Thu Apr  9 16:25:56 2015

@author: ttshimiz
"""

from astropy.modeling import Fittable1DModel, Parameter
import astropy.units as u
import astropy.constants as const
import numpy as np

# Global constants
c_micron = const.c.to(u.micron/u.s).value
Msun_to_g = const.M_sun.to(u.g).value
planck_h = const.h.cgs.value
k_b = const.k_B.cgs.value
c = const.c.cgs.value
pc2cm = const.pc.cgs.value


# Single Temperature Greybody
class Greybody(Fittable1DModel):
    """
    Single temperature greybody.
    Free parameters include dust mass, emissivity, and dust temperature.
    Fixed parameters include kappa_0, lambda_norm, and lumD.

    Equation:
    S_grey = mdust*kappa0*(nu/(c/lamdda_norm))^beta*B(nu,tdust)/lumD^2

    Parameters:
        mdust: log of dust mass in M_sun
        kappa0: normalization of the absorption coefficient
                Default is 0.192 m^2/kg
        lambda_norm: wavelength at which the absorption coefficient
                     is normalized
                     Default is 350 micron.
        beta: dust emissivity
              Default is 2.0.
        tdust: dust temperature in Kelvin
        lumD: luminosity distance in Mpc
        B(nu,tdust): is the standard blackbody equation
    """

    mdust = Parameter()
    tdust = Parameter()
    beta = Parameter(default=2.0, bounds=(0, 5))

    def __init__(self, mdust, tdust, beta, lumD=1.0, kappa0=0.192,
                 lambda_norm=350.):

        self.set_kappa0(kappa0)
        self.set_lambda_norm(lambda_norm)
        self.set_lumD(lumD)

        super(Greybody, self).__init__(mdust, tdust, beta)

    def evaluate(self, x, mdust, tdust, beta):

        md = (10**mdust)*Msun_to_g  # Convert dust mass to grams
        n = c_micron/x  # Calculate frequency of lambda_norm
        bnu = ((2 * planck_h * n**3 / c**2) *
               1 / (np.exp(planck_h * n / k_b / tdust) - 1))
        return (md * self.k0_cgs * (n / self.nu_norm)**beta * np.pi *
                bnu / self.lumD_cm**2)*1e23

    def set_kappa0(self, k0):
        """
        Method to change the normalization of the absorption coefficient
        """

        self.kappa0 = k0*u.m**2/u.kg  # Attach units to kappa_0
        self.k0_cgs = (k0*u.m**2/u.kg).cgs.value   # Convert to cm**2/g

    def set_lambda_norm(self, lnorm):
        """
        Method to change the wavelength at which the absorption is
        normalized.
        """

        self.lambda_norm = lnorm*u.micron  # Attach units to lambda_norm
        self.nu_norm = (c.c.to(u.micron/u.s)/lnorm).value  # Convert to Hz

    def set_lumD(self, ld):
        """
        Method to change the luminosity distance.
        """

        self.lumD = ld*u.Mpc  # Attach units to luminosity distance
        self.lumD_cm = (ld*u.Mpc).cgs.value  # Convert to cm


class TwoTempGreybody(Fittable1DModel):
    """
    Two temperature greybody model with a warm and cold component.

    Equation:
    S_warm = mdust_warm*kappa0*(nu/(c/lamdda_norm))^beta_warm *
             B(nu,tdust_warm)/lumD^2
    S_cold = mdust_cold*kappa0*(nu/(c/lamdda_norm))^beta_cold *
             B(nu,tdust_cold)/lumD^2

    Parameters:
        mdust_warm(cold): log of dust mass for warm (cold) component in M_sun
        kappa0: normalization of the absorption coefficient
                Default is 0.192 m^2/kg. Assumed to be the same for both
                temperature components.
        lambda_norm: wavelength at which the absorption coefficient
                     is normalized
                     Default is 350 micron. Assumed to be the same for both
                     temperature components.
        beta_warm(cold): dust emissivity for the warm (cold) component
                         Default is 2.0 and constrained between 0 and 5.
        tdust_warm(cold): dust temperature in Kelvin for warm (cold) component.
                          tdust_warm is constrained between 35 and 80 K.
                          tdust_cold is constrained between 0 and 35 K.
        lumD: luminosity distance in Mpc
        B(nu,tdust_warm(cold)): is the standard blackbody equation
    """
    # Warm temperature component
    mdust_warm = Parameter()
    tdust_warm = Parameter(bounds=(35, 80))
    beta_warm = Parameter(default=2.0, bounds=(0, 5))

    # Cold temperature component
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

        mwarm = (10**mdust_warm)*Msun_to_g
        mcold = (10**mdust_cold)*Msun_to_g
        n = c_micron/x
        bnu_warm = ((2 * planck_h * n**3 / c**2) *
                    1 / (np.exp(planck_h * n / k_b / tdust_warm) - 1))
        bnu_cold = ((2 * planck_h * n**3 / c**2) *
                    1 / (np.exp(planck_h * n / k_b / tdust_cold) - 1))
        flux_warm = (mwarm * self.k0_cgs *
                     (n / self.nu_norm)**beta_warm *
                     np.pi * bnu_warm /
                     self.lumD_cm**2)*1e23

        flux_cold = (mcold * self.k0_cgs *
                     (n / self.nu_norm)**beta_cold *
                     np.pi * bnu_cold /
                     self.lumD_cm**2)*1e23
        return flux_warm + flux_cold

    def eval_warm(self, x):
        """
        Method for evaluating only the warm component flux.
        """

        mwarm = (10**self.mdust_warm)*Msun_to_g
        n = c_micron/x
        bnu = ((2 * planck_h * n**3 / c**2) *
               1 / (np.exp(planck_h * n / k_b / self.tdust_warm) - 1))
        flux_warm = (mwarm * self.k0_cgs *
                     (n / self.nu_norm)**self.beta_warm *
                     np.pi * bnu /
                     self.lumD_cm**2)*1e23

        return flux_warm

    def eval_cold(self, x):
        """
        Method for evaluating only the cold component flux.
        """

        mcold = (10**self.mdust_cold)*Msun_to_g
        n = c_micron/x
        bnu = ((2 * planck_h * n**3 / c**2) *
               1 / (np.exp(planck_h * n / k_b / self.tdust_cold) - 1))
        flux_cold = (mcold * self.k0_cgs *
                     (n / self.nu_norm)**self.beta_cold *
                     np.pi * bnu /
                     self.lumD_cm**2)*1e23

        return flux_cold

    def set_kappa0(self, k0):
        """
        Method to change the normalization of the absorption coefficient
        """

        self.kappa0 = k0*u.m**2/u.kg  # Attach units to kappa_0
        self.k0_cgs = (k0*u.m**2/u.kg).cgs.value   # Convert to cm**2/g

    def set_lambda_norm(self, lnorm):
        """
        Method to change the wavelength at which the absorption is
        normalized.
        """

        self.lambda_norm = lnorm*u.micron  # Attach units to lambda_norm
        self.nu_norm = (c.c.to(u.micron/u.s)/lnorm).value  # Convert to Hz

    def set_lumD(self, ld):
        """
        Method to change the luminosity distance.
        """

        self.lumD = ld*u.Mpc  # Attach units to luminosity distance
        self.lumD_cm = (ld*u.Mpc).cgs.value  # Convert to cm


class GreybodyPowerlaw(Fittable1DModel):
    """
    Greybody+Powerlaw model described in Casey 2012

    Equation:
    S_grey = mdust*kappa0*(nu/(c/lamdda_norm))^beta*B(nu,tdust)/lumD^2
    S_pwlw = pownorm * (lambda/wturn)^alpha * exp(-(lambda/wturn)^2)
    S_model = S_grey + S_pwlw

    Parameters:
        mdust: log of dust mass in M_sun
        kappa0: normalization of the absorption coefficient
                Default is 0.192 m^2/kg and is fixed during fitting.
        lambda_norm: wavelength at which the absorption coefficient
                     is normalized
                     Default is 350 micron and is fixed during fitting.
        beta: dust emissivity
              Default is 2.0.
        tdust: dust temperature in Kelvin
        lumD: luminosity distance in Mpc and is fixed during fitting.
        B(nu,tdust): is the standard blackbody equation
        pownorm: log of the normalization of the powerlaw component
        alpha: powerlaw slope. By default constrained to be between -5 and 5.
        wturn: wavelength where the powerlaw component is exponentially cutoff.
               By default constrained to be between 20 and 100.
    """
    mdust = Parameter()
    tdust = Parameter()
    beta = Parameter(default=2.0, bounds=(0, 5))
    pownorm = Parameter()
    alpha = Parameter(default=2.0, bounds=(-5, 5))
    wturn = Parameter(default=40.0, bounds=(20, 100))

    def __init__(self, mdust, tdust, beta, pownorm, alpha, wturn,
                 kappa0=0.192, lambda_norm=350., redshift=0, lumD=1.0):

        self.set_kappa0(kappa0)
        self.set_lambda_norm(lambda_norm)
        self.set_lumD(lumD)
        self.set_redshift(redshift)

        super(GreybodyPowerlaw, self).__init__(mdust, tdust, beta,
                                               pownorm, alpha, wturn)

    def evaluate(self, x, mdust, tdust, beta,
                 pownorm, alpha, wturn):

        md = (10**mdust)*Msun_to_g
        n = c_micron/x
        bnu = ((2 * planck_h * n**3 / c**2) *
               1 / (np.exp(planck_h * n / k_b / tdust) - 1))
        flux_grey = (md * self.k0_cgs *
                     (n / self.nu_norm)**beta *
                     np.pi * bnu /
                     self.lumD_cm**2)*1e23

        flux_plaw = (10**pownorm) * (x/wturn)**alpha * np.exp(-(x/wturn)**2)

        return flux_grey + flux_plaw

    def eval_grey(self, x):
        """
        Method to calculate only the greybody component flux.

        Input
        -----
        x: scalar or array
           The wavelengths in micron to calculate the flux

        Output
        ------
        flux: scalar or array
              The flux at wavelengths, x in Jy
        """

        md = (10**self.mdust)*Msun_to_g
        n = c_micron/x
        bnu = ((2 * planck_h * n**3 / c**2) *
               1 / (np.exp(planck_h * n / k_b / self.tdust) - 1))
        flux_grey = (md * self.k0_cgs *
                     (n / self.nu_norm)**self.beta *
                     np.pi * bnu /
                     self.lumD_cm**2)*1e23
        return flux_grey

    def eval_plaw(self, x):
        """
        Method to calculate only the powerlaw component flux.

        Input
        -----
        x: scalar or array
           The wavelengths in micron to calculate the flux

        Output
        ------
        flux: scalar or array
              The flux at wavelengths, x in Jy
        """

        return ((10**self.pownorm) * (x/self.wturn)**self.alpha *
                np.exp(-(x/self.wturn)**2))

    def set_kappa0(self, k0):
        """
        Method to change the normalization of the absorption coefficient
        """

        self.kappa0 = k0*u.m**2/u.kg  # Attach units to kappa_0
        self.k0_cgs = (k0*u.m**2/u.kg).cgs.value   # Convert to cm**2/g

    def set_lambda_norm(self, lnorm):
        """
        Method to change the wavelength at which the absorption is
        normalized.
        """

        self.lambda_norm = lnorm*u.micron  # Attach units to lambda_norm
        self.nu_norm = (c_micron/lnorm)  # Convert to Hz

    def set_lumD(self, ld):
        """
        Method to change the luminosity distance.
        """

        self.lumD = ld*u.Mpc  # Attach units to luminosity distance
        self.lumD_cm = (ld*u.Mpc).cgs.value  # Convert to cm

    def set_redshift(self, zz):
        """
        Method to change the redshift.
        """

        self.redshift = zz
