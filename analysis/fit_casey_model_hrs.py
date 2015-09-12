# -*- coding: utf-8 -*-
"""
Created on Tue Sept 08 14:42:19 2015

@author: ttshimiz
"""

import sys
sys.path.append('../../bat-agn-sed-fitting')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import plotting as bat_plot
import fitting as bat_fit
import models as bat_model
import pandas as pd
from astropy.modeling import fitting as apy_fit
import pickle
import matplotlib.pyplot as plt

# Upload the WISE and Herschel fluxes for HRS
hrs_data = pd.read_table('../../spire-catalog-analysis/hrs-data/hrs_photometry.txt', index_col=0)
hrs_sed = pd.DataFrame({'W3': hrs_data['S12'], 'W4': hrs_data['S22'],
                        'PACS100': hrs_data['S100']*1000, 'PACS160': hrs_data['S160']*1000,
                        'PSW': hrs_data['S250'], 'PMW': hrs_data['S350'],
                        'PLW': hrs_data['S500']})

hrs_err = pd.DataFrame({'W3_err': hrs_data['err12'], 'W4_err': hrs_data['err22'],
                        'PACS100_err': hrs_data['err100']*1000, 'PACS160_err': hrs_data['err160']*1000,
                        'PSW_err': hrs_data['err_tot250'], 'PMW_err': hrs_data['err_tot350'],
                        'PLW_err': hrs_data['err_tot500']})
hrs_flag = hrs_data[['Flag12', 'Flag22', 'Flag100', 'Flag160', 'Flag250', 'Flag350', 'Flag500']]

hrs_err.loc[hrs_flag['Flag12'] == 0, 'W3_err'] = 5./3.*hrs_err.loc[hrs_flag['Flag12'] == 0, 'W3_err']
hrs_err.loc[hrs_flag['Flag22'] == 0, 'W4_err'] = 5./3.*hrs_err.loc[hrs_flag['Flag22'] == 0, 'W4_err']
hrs_err.loc[hrs_flag['Flag100'] == 0, 'PACS100_err'] = 5./3.*hrs_sed.loc[hrs_flag['Flag100'] == 0, 'PACS100']
hrs_err.loc[hrs_flag['Flag160'] == 0, 'PACS160_err'] = 5./3.*hrs_sed.loc[hrs_flag['Flag160'] == 0, 'PACS160']
hrs_err.loc[hrs_flag['Flag250'] == 0, 'PSW_err'] = 5./3.*hrs_sed.loc[hrs_flag['Flag250'] == 0, 'PSW']
hrs_err.loc[hrs_flag['Flag350'] == 0, 'PMW_err'] = 5./3.*hrs_sed.loc[hrs_flag['Flag350'] == 0, 'PMW']
hrs_err.loc[hrs_flag['Flag500'] == 0, 'PLW_err'] = 5./3.*hrs_sed.loc[hrs_flag['Flag500'] == 0, 'PLW']

#hrs_sed.loc[hrs_flag['Flag12'] == 0, 'W3'] = np.nan
#hrs_sed.loc[hrs_flag['Flag22'] == 0, 'W4'] = np.nan
hrs_sed.loc[hrs_flag['Flag100'] == 0, 'PACS100'] = np.nan
hrs_sed.loc[hrs_flag['Flag160'] == 0, 'PACS160'] = np.nan
hrs_sed.loc[hrs_flag['Flag250'] == 0, 'PSW'] = np.nan
hrs_sed.loc[hrs_flag['Flag350'] == 0, 'PMW'] = np.nan
hrs_sed.loc[hrs_flag['Flag500'] == 0, 'PLW'] = np.nan

# Upper limits in the HRS photometry were all 3 sigma upper limits
# We need to change the data to a 5-sigma cutoff for all of the photometry
#hrs_flag.loc[hrs_sed['W3']/hrs_err['W3_err'] < 5, 'Flag12'] = 0
#hrs_err.loc[(hrs_sed['W3']/hrs_err['W3_err'] < 5), 'W3_err'] = 5.*hrs_err.loc[(hrs_sed['W3']/hrs_err['W3_err'] < 5) & (hrs_sed['W3'] != 0), 'W3_err']
#hrs_sed.loc[hrs_sed['W3']/hrs_err['W3_err'] < 5, 'W3'] = np.nan

#hrs_flag.loc[hrs_sed['W4']/hrs_err['W4_err'] < 5, 'Flag22'] = 0
#hrs_err.loc[(hrs_sed['W4']/hrs_err['W4_err'] < 5), 'W4_err'] = 5.*hrs_err.loc[(hrs_sed['W4']/hrs_err['W4_err'] < 5) & (hrs_sed['W4'] != 0), 'W4_err']
#hrs_sed.loc[hrs_sed['W4']/hrs_err['W4_err'] < 5, 'W4'] = np.nan

hrs_flag.loc[hrs_sed['PACS100']/hrs_err['PACS100_err'] < 5, 'Flag100'] = 0
hrs_err.loc[(hrs_sed['PACS100']/hrs_err['PACS100_err'] < 5), 'PACS100_err'] = 5.*hrs_err.loc[(hrs_sed['PACS100']/hrs_err['PACS100_err'] < 5) & (hrs_sed['PACS100'] != 0), 'PACS100_err']
hrs_sed.loc[hrs_sed['PACS100']/hrs_err['PACS100_err'] < 5, 'PACS100'] = np.nan

hrs_flag.loc[hrs_sed['PACS160']/hrs_err['PACS160_err'] < 5, 'Flag160'] = 0
hrs_err.loc[(hrs_sed['PACS160']/hrs_err['PACS160_err'] < 5), 'PACS160_err'] = 5.*hrs_err.loc[(hrs_sed['PACS160']/hrs_err['PACS160_err'] < 5) & (hrs_sed['PACS160'] != 0), 'PACS160_err']
hrs_sed.loc[hrs_sed['PACS160']/hrs_err['PACS160_err'] < 5, 'PACS160'] = np.nan

hrs_flag.loc[hrs_sed['PSW']/hrs_err['PSW_err'] < 5, 'Flag250'] = 0
hrs_err.loc[(hrs_sed['PSW']/hrs_err['PSW_err'] < 5), 'PSW_err'] = 5.*hrs_err.loc[(hrs_sed['PSW']/hrs_err['PSW_err'] < 5) & (hrs_sed['PSW'] != 0), 'PSW_err']
hrs_sed.loc[hrs_sed['PSW']/hrs_err['PSW_err'] < 5, 'PSW'] = np.nan

hrs_flag.loc[hrs_sed['PMW']/hrs_err['PMW_err'] < 5, 'Flag350'] = 0
hrs_err.loc[(hrs_sed['PMW']/hrs_err['PMW_err'] < 5), 'PMW_err'] = 5.*hrs_err.loc[(hrs_sed['PMW']/hrs_err['PMW_err'] < 5) & (hrs_sed['PMW'] != 0), 'PMW_err']
hrs_sed.loc[hrs_sed['PMW']/hrs_err['PMW_err'] < 5, 'PMW'] = np.nan

hrs_flag.loc[hrs_sed['PLW']/hrs_err['PLW_err'] < 5, 'Flag500'] = 0
hrs_err.loc[(hrs_sed['PLW']/hrs_err['PLW_err'] < 5), 'PLW_err'] = 5.*hrs_err.loc[(hrs_sed['PLW']/hrs_err['PLW_err'] < 5) & (hrs_sed['PLW'] != 0), 'PLW_err']
hrs_sed.loc[hrs_sed['PLW']/hrs_err['PLW_err'] < 5, 'PLW'] = np.nan


hrs_detect = pd.DataFrame({'W3_Detect': (hrs_flag['Flag12'] == 1) | (hrs_flag['Flag12'] == 2),
                           'W4_Detect': (hrs_flag['Flag22'] == 1) | (hrs_flag['Flag22'] == 2),
                           'PACS100_Detect': (hrs_flag['Flag100'] == 1) | (hrs_flag['Flag100'] == 2),
                           'PACS160_Detect': (hrs_flag['Flag160'] == 1) | (hrs_flag['Flag160'] == 2),
                           'PSW_Detect': (hrs_flag['Flag250'] == 1) | (hrs_flag['Flag250'] == 2),
                           'PMW_Detect': (hrs_flag['Flag350'] == 1) | (hrs_flag['Flag350'] == 2),
                           'PLW_Detect': (hrs_flag['Flag500'] == 1) | (hrs_flag['Flag500'] == 2)})

nbands_detect = hrs_detect.sum(axis=1)

filters = np.array(['W3', 'W4', 'PACS100', 'PACS160', 'PSW', 'PMW', 'PLW'])
filt_err = np.array([s+'_err' for s in filters])
waves = np.array([12., 22., 100., 160., 250., 350., 500.])

# Uncomment to fit sources with detections at all wavelengths
# sed_use = sed.dropna(how='any')

# Uncomment to fit sources with only N detected points.
# Change the integer on the right side of '==' to N.
src_use = ((nbands_detect >= 4) & np.isfinite(hrs_sed['W3']) & np.isfinite(hrs_sed['W4']))
sed_use = hrs_sed[src_use]
err_use = hrs_err[src_use]
names_use = sed_use.index

base_model = bat_model.GreybodyPowerlaw(0.0, 25., 2.0, 0.0, 2.0, 45.0)
lev_marq = apy_fit.LevMarLSQFitter()
bayes = bat_fit.SEDBayesFitter(threads=8)

# Fix parameters
#base_model.wturn.fixed = True
base_model.beta.fixed = True

#for n in names_use:
for n in [1]:    
    print 'Fitting: HRS ', n
    src_sed = sed_use.loc[n][filters]/1000.
    src_err = err_use.loc[n][filt_err]/1000.
    
    flux = np.array(src_sed, dtype=np.float)
    flux_err = np.array(src_err, dtype=np.float)
    
    flux_detected = np.isfinite(flux)
    flux_use = flux_err != 0
    
    flux_good = flux[flux_detected & flux_use]
    flux_err_good = flux_err[flux_detected & flux_use]
    filts_good = filters[flux_detected & flux_use]
    waves_good = waves[flux_detected & flux_use]
	
    src_lumD = hrs_data.loc[n]['Dist_Mpc']

    model_ml = base_model.copy()
    model_ml.set_lumD(src_lumD)
    model_ml.set_redshift(None)
      
    # Roughly guess initial parameters using the data
    # Use the slope between W4 and W3 as guess for alpha
    if (np.isfinite(src_sed['W3'])) & (np.isfinite(src_sed['W4'])):
        alpha_init = (np.log10(src_sed['W4']/src_sed['W3']) /
                  	  np.log10(waves[1]/waves[0]))
        model_ml.alpha.value = alpha_init
    else:
        model_ml.alpha.value = 1.0
    
    # Use the flux density at 250, 160, 350, or 100 as guess for Mdust
    if not np.isnan(src_sed['PSW']):
        mdust_init = np.log10(src_sed['PSW']/model_ml.eval_grey(250.))
    elif not np.isnan(src_sed['PACS160']):
        mdust_init = np.log10(src_sed['PACS160']/model_ml.eval_grey(160.))
    elif not np.isnan(src_sed['PMW']):
        mdust_init = np.log10(src_sed['PMW']/model_ml.eval_grey(350.))
    elif not np.isnan(src_sed['PACS100']):
        mdust_init = np.log10(src_sed['PACS100']/model_ml.eval_grey(70.))
    model_ml.mdust.value = mdust_init
    
    # Use the W3 flux density as guess for normalization of powerlaw
    if np.isfinite(src_sed['W3']):
        pownorm_init = np.log10(src_sed['W3']/model_ml.eval_plaw(12))
    elif np.isfinite(src_sed['W4']):
        pownorm_init = np.log10(src_sed['W3']/model_ml.eval_plaw(22))
    else:
        pownorm_init = 0.0
    model_ml.pownorm.value = pownorm_init
    
    # Fix certain parameters for the maximum likelihood estimate
    model_ml.wturn.fixed = True
    model_ml.beta.fixed = True
    model_ml.tdust.fixed = True
    
    # Fit the model using Levenberg-Marquardt algorithm to get the best initial guesses
    model_init = lev_marq(model_ml, waves_good, flux_good, weights=1/flux_err_good,
                          maxiter=1000)
    
    # Change back the fixed parameters to what's wanted for Bayesian analysis
    model_init.wturn.fixed = False
    model_init.beta.fixed = True
    model_init.tdust.fixed = False
    model_init.mdust.value = 5.0
    
    # Fit the model using MCMC
    model_final = bayes.fit(model_init, waves[flux_use], flux[flux_use], yerr=flux_err[flux_use], use_nondetect=True,
                            filts=filters[flux_use])

    # Plot the best fit along with the data
    print 'Plotting the fit: ', n
    fig_fit = bat_plot.plot_fit(waves[flux_use], flux[flux_use], model_final, obs_err=flux_err[flux_use],
                                plot_components=True, plot_mono_fluxes=True,
                                filts=filters[flux_use], plot_fit_spread=True,
                                name=n, plot_params=True)
    fig_fit.savefig('casey_bayes_results/hrs_beta_fixed_2_wturn_gaussianPrior/sed_plots/HRS' + str(n) +
                    '_casey_bayes_beta_fixed_2_wturn_gaussianPrior_sed_fit.png',
                    bbox_inches='tight')
    plt.close(fig_fit)

    # Plot the marginal posterior distributions for each fitted parameter
    fig_triangle = bat_plot.plot_triangle(model_final)
    fig_triangle.savefig('casey_bayes_results/hrs_beta_fixed_2_wturn_gaussianPrior/triangle_plots/HRS' + str(n) +
                         '_casey_bayes_beta_fixed_2_wturn_gaussianPrior_triangle.png',
                         bbox_inches='tight')
    plt.close(fig_triangle)

    # Save the fitting results in a Python pickle file
    print 'Saving the fit: ', n
    fit_dict = {'name': n,
                'flux': flux,
                'flux_err': flux_err,
                'best_fit_model': model_final,
                'filters': filters,
                'waves': waves}
    pickle_file = open('casey_bayes_results/hrs_beta_fixed_2_wturn_gaussianPrior/pickles/HRS' + str(n) +
                       '_casey_bayes_beta_fixed_2_wturn_gaussianPrior.pickle', 'wb')
    pickle.dump(fit_dict, pickle_file)
    pickle_file.close()
