# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:10:19 2015

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

# Upload the BAT fluxes for Herschel and WISE
herschel_data = pd.read_csv('../../bat-data/bat_herschel.csv', index_col=0,
                            na_values=0)
wise_data = pd.read_csv('../../bat-data/bat_wise.csv', index_col=0,
                        usecols=[0, 1, 2, 4, 5, 7, 8, 10, 11], na_values=0)

sed = herschel_data.join(wise_data[['W3', 'W3_err', 'W4', 'W4_err']])

# SPIRE fluxes that are seriously contaminated by a companion should be upper limits
psw_flag = herschel_data['PSW_flag']
pmw_flag = herschel_data['PMW_flag']
plw_flag = herschel_data['PLW_flag']

sed['PSW_err'][psw_flag == 'AD'] = sed['PSW'][psw_flag == 'AD']
sed['PSW'][psw_flag == 'AD'] = np.nan
sed['PMW_err'][pmw_flag == 'AD'] = sed['PMW'][pmw_flag == 'AD']
sed['PMW'][pmw_flag == 'AD'] = np.nan
sed['PLW_err'][plw_flag == 'AD'] = sed['PLW'][plw_flag == 'AD']
sed['PLW'][plw_flag == 'AD'] = np.nan

# Upload info on BAT AGN for redshift and luminosity distance
bat_info = pd.read_csv('../../bat-data/bat_info.csv', index_col=0)

filt_use = np.array(['W3', 'W4', 'PACS70', 'PACS160', 'PSW', 'PMW', 'PLW'])
filt_err = np.array([s+'_err' for s in filt_use])
waves = np.array([12., 22., 70., 160., 250., 350., 500.])

# Uncomment to fit sources with detections at all wavelengths
# sed_use = sed.dropna(how='any')

# Uncomment to fit sources with only N detected points.
# Change the integer on the right side of '==' to N.

sed_use = sed[np.sum(np.isfinite(sed[filt_use].values), axis=1) >= 6]

names_use = sed_use.index

base_model = bat_model.GreybodyPowerlaw(0.0, 25., 2.0, 0.0, 2.0, 50.0)
lev_marq = apy_fit.LevMarLSQFitter()
bayes = bat_fit.SEDBayesFitter(threads=8)

# Fix parameters
#base_model.wturn.fixed = True
base_model.beta.fixed = True

for n in names_use:
#for n in names_use[1:2]:
    print 'Fitting: ', n
    src_sed = sed_use.loc[n][filt_use]
    flux = np.array(src_sed, dtype=np.float)
    flux_detected = np.isfinite(flux)
    flux_use = flux[flux_detected]
    src_err = sed_use.loc[n][filt_err]
    flux_err = np.array(src_err, dtype=np.float)
    flux_err_use = flux_err[flux_detected]
    filt_detected = filt_use[flux_detected]
    waves_use = waves[flux_detected]
	
    src_z = bat_info.loc[n]['Redshift']
    src_lumD = bat_info.loc[n]['Dist_[Mpc]']

    model_ml = base_model.copy()
    model_ml.set_redshift(src_z)
    model_ml.set_lumD(src_lumD)

    alpha_init = (np.log10(src_sed['W4']/src_sed['W3']) /
                  np.log10(waves[1]/waves[0]))
    model_ml.alpha.value = alpha_init

    mdust_init = np.log10(src_sed['PSW']/model_ml.eval_grey(250))
    model_ml.mdust.value = mdust_init

    pownorm_init = np.log10(src_sed['W3']/model_ml.eval_plaw(12))
    model_ml.pownorm.value = pownorm_init

    model_init = lev_marq(model_ml, waves_use, flux_use, weights=1/flux_err_use,
                          maxiter=1000)
    
    model_final = bayes.fit(model_init, waves, flux, yerr=flux_err, use_nondetect=True
                            filts=filt_use)
    print 'Plotting the fit: ', n
    fig_fit = bat_plot.plot_fit(waves, flux, model_final, obs_err=flux_err,
                                plot_components=True, plot_mono_fluxes=True,
                                filts=filt_use, plot_fit_spread=True,
                                name=n, plot_params=True)
    fig_fit.savefig('casey_bayes_results/beta_fixed_2/sed_plots/' + n +
                    '_casey_bayes_beta_fixed_2_sed_fit.png',
                    bbox_inches='tight')
    plt.close(fig_fit)
    fig_triangle = bat_plot.plot_triangle(model_final)
    fig_triangle.savefig('casey_bayes_results/beta_fixed_2/triangle_plots/' + n +
                         '_casey_bayes_beta_fixed_2_triangle.png',
                         bbox_inches='tight')
    plt.close(fig_triangle)
    print 'Saving the fit: ', n
    fit_dict = {'name': n,
                'flux': flux,
                'flux_err': flux_err,
                'best_fit_model': model_final,
                'filters': filt_use,
                'waves': waves}
    pickle_file = open('casey_bayes_results/beta_fixed_2/pickles/' + n +
                       '_casey_bayes_beta_fixed_2.pickle', 'wb')
    pickle.dump(fit_dict, pickle_file)
    pickle_file.close()
