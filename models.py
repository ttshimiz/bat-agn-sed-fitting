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

Created on Thu Apr	9 16:25:56 2015

@author: ttshimiz
"""

from astropy.modeling import Fittable1DModel, Parameter
import astropy.cosmology as apy_cosmo
import astropy.units as u
import astropy.constants as const
import numpy as np
from scipy import optimize as opt
from fitting import Filters

# Global constants
c_micron = const.c.to(u.micron/u.s).value
Msun_to_g = const.M_sun.to(u.g).value
planck_h = const.h.cgs.value
k_b = const.k_B.cgs.value
c = const.c.cgs.value
pc2cm = const.pc.cgs.value

# Set the cosmology to use
cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# Import all of the Filters
filters = Filters()


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

	def __init__(self, mdust, tdust, beta, kappa0=0.192,
				 lambda_norm=350., lumD=None, redshift=0.001):

		self.set_kappa0(kappa0)
		self.set_lambda_norm(lambda_norm)
		self.set_redshift(redshift)
		self.set_lumD(lumD)

		super(Greybody, self).__init__(mdust, tdust, beta)

	def evaluate(self, x, mdust, tdust, beta):

		md = (10**mdust)*Msun_to_g	# Convert dust mass to grams
		n = c_micron/x	# Calculate frequency of lambda_norm
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
		self.nu_norm = (c_micron/lnorm)	 # Convert to Hz
	
	def set_lumD(self, ld):
		"""
		Method to change the luminosity distance.
		"""
		if ld is None:
			self.lumD = cosmo.luminosity_distance(self.redshift).value
			self.lumD_cm = cosmo.luminosity_distance(self.redshift).cgs.value
		else:
			self.lumD = ld	# Attach units to luminosity distance
			self.lumD_cm = (ld*u.Mpc).cgs.value	 # Convert to cm

	def set_redshift(self, zz):
		"""
		Method to change the redshift.
		"""

		self.redshift = zz

	def calc_luminosity(self, lower=8.0, upper=1000.0):

		waves = np.arange(lower, upper, 0.01)
		freq = c_micron/waves

		flux_density = self(waves)/1e23
		integral = -np.trapz(flux_density, freq)
		lum = (4*np.pi)*(self.lumD_cm)**2*integral/const.L_sun.cgs.value

		return lum


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
				 kappa0=0.192, lambda_norm=350.,
				 lumD=None, redshift=0.001):

		self.set_kappa0(kappa0)
		self.set_lambda_norm(lambda_norm)
		self.set_redshift(redshift)
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
		if ld is None:
			self.lumD = cosmo.luminosity_distance(self.redshift).value
			self.lumD_cm = cosmo.luminosity_distance(self.redshift).cgs.value
		else:
			self.lumD = ld	# Attach units to luminosity distance
			self.lumD_cm = (ld*u.Mpc).cgs.value	 # Convert to cm

	def set_redshift(self, zz):
		"""
		Method to change the redshift.
		"""

		self.redshift = zz

	def calc_luminosity(self, lower=8.0, upper=1000.0):

		waves = np.arange(lower, upper, 0.01)
		freq = c_micron/waves

		flux_density = self(waves)/1e23
		integral = -np.trapz(flux_density, freq)
		lum = (4*np.pi)*(self.lumD_cm)**2*integral/const.L_sun.cgs.value

		return lum

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
	mdust = Parameter(bounds=(1, 10))
	tdust = Parameter(bounds=(0, 100))
	beta = Parameter(default=2.0, bounds=(0, 5))
	pownorm = Parameter(bounds=(-10, 10))
	alpha = Parameter(default=2.0, bounds=(-5, 5))
	wturn = Parameter(default=40.0, bounds=(20, 100))

	def __init__(self, mdust, tdust, beta, pownorm, alpha, wturn,
				 kappa0=0.192, lambda_norm=350., redshift=0.001, lumD=None):

		self.set_kappa0(kappa0)
		self.set_lambda_norm(lambda_norm)
		self.set_redshift(redshift)
		self.set_lumD(lumD)
		self.chain = None
		self.chain_nb = None
		self.param_errs = None

		self.n_components = 2
		self.comp_names = ['Greybody', 'Exp. Powerlaw']

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

	def eval_comps(self, x):

		comp1 = self.eval_grey(x)
		comp2 = self.eval_plaw(x)

		return np.vstack([comp1, comp2])

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
		self.nu_norm = (c_micron/lnorm)	 # Convert to Hz

	def set_lumD(self, ld):
		"""
		Method to change the luminosity distance.
		"""
		if ld is None:
			self.lumD = cosmo.luminosity_distance(self.redshift).value
			self.lumD_cm = cosmo.luminosity_distance(self.redshift).cgs.value
		else:
			self.lumD = ld	# Attach units to luminosity distance
			self.lumD_cm = (ld*u.Mpc).cgs.value	 # Convert to cm

	def set_redshift(self, zz):
		"""
		Method to change the redshift.
		"""

		self.redshift = zz

	def calc_luminosity(self, lower=8.0, upper=1000.0):

		waves = np.arange(lower, upper, 0.01)
		freq = c_micron/waves

		flux_density = self(waves)/1e23
		integral = -np.trapz(flux_density, freq)
		lum = (4*np.pi)*(self.lumD_cm)**2*integral/const.L_sun.cgs.value

		return lum


class Dale2014(object):

	def __init__(self, redshift=0.001, lumD=None):

		self.set_redshift(redshift)
		self.set_lumD(lumD)

		self.alpha = np.arange(0.0625, 4.0625, 0.0625)
		self.fracAGN = np.arange(0, 1.05, 0.05)
		self.waves = np.loadtxt('./Dale14/spectra/spectra.0.00AGN.dat',
								usecols=[0])

		self.models = np.zeros((len(self.alpha),
								len(self.waves),
								len(self.fracAGN)))

		for i in range(len(self.fracAGN)):
			fn = 'spectra.{0:0.2f}AGN.dat'.format(self.fracAGN[i])
			m = np.loadtxt('./Dale14/spectra/'+fn).T
			self.models[:, :, i] = m[1:, :]+44 - np.log10(c_micron/self.waves)

	def set_lumD(self, ld):
		"""
		Method to change the luminosity distance.
		"""
		if ld is None:
			self.lumD = cosmo.luminosity_distance(self.redshift).value
			self.lumD_cm = cosmo.luminosity_distance(self.redshift).cgs.value
		else:
			self.lumD = ld	# Attach units to luminosity distance
			self.lumD_cm = (ld*u.Mpc).cgs.value	 # Convert to cm

	def set_redshift(self, zz):
		"""
		Method to change the redshift.
		"""

		self.redshift = zz

	def fit(self, x, y, yerr=None, filts=None,
			alpha_use='all', fracAGN_use='all'):

		if alpha_use == 'all':
			self.alpha_use = np.arange(0.0625, 4.0, 0.0625)
		else:
			self.alpha_use = alpha_use

		if fracAGN_use == 'all':
			self.fracAGN_use = np.arange(0, 1.05, 0.05)
		else:
			self.fracAGN_use = fracAGN_use

		if yerr is None:
			yerr = np.ones(len(x))

		self.chi_sq = np.zeros((len(self.alpha_use), len(self.fracAGN_use)))
		self.norms = np.zeros((len(self.alpha_use), len(self.fracAGN_use)))

		def errfunc(norm, obs, model, err):
			return (obs - (norm*model)) / err
	
		zcorr = (1 + self.redshift)
		for a in range(len(self.alpha_use)):
			for f in range(len(self.fracAGN_use)):

				model_fluxes = np.zeros(len(x))

				for w in range(len(x)):
					sed = self.get_sed(self.alpha_use[a], self.fracAGN[f])
					model_fluxes[w] = filters.calc_mono_flux(filts[w],
															 self.waves*zcorr, sed)
				out = opt.leastsq(errfunc, 1, args=(y, model_fluxes, yerr))

				self.norms[a, f] = out[0][0]
				self.chi_sq[a, f] = np.sum(errfunc(self.norms[a, f], y,
										   model_fluxes, yerr)**2)

		self.best_fit = {}
		indmin = np.where(self.chi_sq == min(self.chi_sq.flatten()))
		self.best_fit['chi2'] = min(self.chi_sq.flatten())
		self.best_fit['alpha'] = self.alpha_use[indmin[0][0]]
		self.best_fit['fracAGN'] = self.fracAGN_use[indmin[1][0]]
		self.best_fit['norm'] = self.norms[indmin[0][0], indmin[1][0]]
		self.best_fit['sed'] = self.get_sed(self.best_fit['alpha'],
											self.best_fit['fracAGN'],
											self.best_fit['norm'])
		lir = self.calc_luminosity(self.best_fit['alpha'],
								   self.best_fit['fracAGN'],
								   self.best_fit['norm'])
		self.best_fit['LIR'] = np.log10(lir)

	def get_sed(self, alph, fa, norm=1.0):
		i = self.alpha == alph
		j = self.fracAGN == fa
		sed = norm*10**self.models[i, :, j].flatten()
		return sed

	def calc_luminosity(self, alph, fa, norm=1.0, low=8., high=1000.):
		sed = self.get_sed(alph, fa, norm=norm)
		waves = self.waves
		freq = c_micron/waves

		flux_density = norm*sed/1e23
		integral = -np.trapz(flux_density, freq)
		lum = (4*np.pi)*(self.lumD_cm)**2*integral/const.L_sun.cgs.value

		return lum

class DecompIR(object):

	def __init__(self, redshift=0.001, lumD=None):

		self.set_redshift(redshift)
		self.set_lumD(lumD)
		
		self.host_names = ['SB1', 'SB2', 'SB3', 'SB4', 'SB5']
		self.agn_names = ['Mean', 'HiLum', 'LoLum']
		
		self.waves = np.loadtxt('./DecompIR/templates.txt', usecols=[0])
		models = np.loadtxt('./DecompIR/templates.txt', usecols=[1, 2, 3, 4, 5, 6, 7, 8])
		self.host_models = {'SB1': models[:, 3],
							'SB2': models[:, 4],
							'SB3': models[:, 5],
							'SB4': models[:, 6],
							'SB5': models[:, 7]}
		self.agn_models = {'Mean': models[:, 0],
						   'HiLum': models[:, 1],
						   'LoLum': models[:, 2]}
	
	def set_lumD(self, ld):
		"""
		Method to change the luminosity distance.
		"""
		if ld is None:
			self.lumD = cosmo.luminosity_distance(self.redshift).value
			self.lumD_cm = cosmo.luminosity_distance(self.redshift).cgs.value
		else:
			self.lumD = ld	# Attach units to luminosity distance
			self.lumD_cm = (ld*u.Mpc).cgs.value	 # Convert to cm

	def set_redshift(self, zz):
		"""
        Method to change the redshift.
        """
		self.redshift = zz
	
	def fit(self, x, y, yerr=None, filts=None,
			agn_use='Mean', host_use='all'):
		
		self.agn_use_name = agn_use
		self.agn_use = self.agn_models[agn_use]

		if host_use == 'all':
			self.host_use = self.host_models
		else:
			self.host_use = {f:host_models[f] for f in host_use}

		if yerr is None:
			yerr = np.ones(len(x))

		self.chi_sq = np.zeros(len(self.host_use))
		self.norms = np.zeros((len(self.host_use), 2))

		def errfunc(params, obs, agn_model, host_model, err):
			norm_agn = params[0]
			norm_host = params[1]
			return (obs - (10**norm_agn*agn_model + 10**norm_host*host_model)) / err

		for h in range(len(self.host_use)):

			agn_model_fluxes = np.zeros(len(x))
			host_model_fluxes = np.zeros(len(x))
			host_sed = self.host_use[self.host_use.keys()[h]]
			zcorr = (1+self.redshift)
			
			for w in range(len(x)):
				agn_model_fluxes[w] = filters.calc_mono_flux(filts[w],
														     self.waves*zcorr, self.agn_use)
				host_model_fluxes[w] = filters.calc_mono_flux(filts[w],
				                                              self.waves*zcorr, host_sed)
			out = opt.leastsq(errfunc, [0, 0], args=(y, agn_model_fluxes,
			                                         host_model_fluxes,
			                                         yerr))

			self.norms[h, 0] = out[0][0]
			self.norms[h, 1] = out[0][1]
			self.chi_sq[h] = np.sum(errfunc(self.norms[h], y,
								    agn_model_fluxes, host_model_fluxes, yerr)**2)

		self.best_fit = {}
		indmin = np.argmin(self.chi_sq)
		self.best_fit['chi2'] = min(self.chi_sq)
		self.best_fit['host_name'] = self.host_use.keys()[indmin]
		self.best_fit['agn_name'] = self.agn_use_name
		self.best_fit['norms'] = self.norms[indmin]
		self.best_fit['total_sed'] = self.get_sed(self.best_fit['agn_name'],
											self.best_fit['host_name'],
											self.best_fit['norms'])
		self.best_fit['agn_sed'] = 10**self.best_fit['norms'][0]*self.agn_use
		self.best_fit['host_sed'] = (10**self.best_fit['norms'][1]*
		                            self.host_models[self.best_fit['host_name']])
		lir = self.calc_luminosity(self.best_fit['agn_name'],
								   self.best_fit['host_name'],
								   self.best_fit['norms'])
		self.best_fit['LIR'] = np.log10(lir)
		self.best_fit['fracAGN'] = (self.calc_luminosity(self.best_fit['agn_name'],
		                                                self.best_fit['host_name'],
		                                                [self.best_fit['norms'][0], -100]) /
		                            lir)
	
	def get_sed(self, agn_name, host_name, norms=[0, 0]):
		return (10**norms[0]*self.agn_models[agn_name] +
		        10**norms[1]*self.host_models[host_name])

	def calc_luminosity(self, agn_name, host_name, norms=[0,0], low=8., high=1000.):
		sed = self.get_sed(agn_name, host_name, norms=norms)
		waves = self.waves
		freq = c_micron/waves

		flux_density = sed/1e23
		integral = -np.trapz(flux_density, freq)
		lum = (4*np.pi)*(self.lumD_cm)**2*integral/const.L_sun.cgs.value

		return lum