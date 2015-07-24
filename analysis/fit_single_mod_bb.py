# -*- coding: utf-8 -*-
"""
Created on Sat Jul 07 08:53:30 2015

@author: ttshimiz
"""

import sys
sys.path.append('../../bat-agn-sed-fitting/')

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

sed = herschel_data[['PACS70', 'PACS70_err',
                     'PACS160', 'PACS160_err',
                     'PSW', 'PSW_err',
                     'PMW', 'PMW_err',
                     'PLW', 'PLW_err']]

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

# For single modified blackbody fitting, use the 70 um point as an upper limit
pacs70_chng = np.isfinite(sed['PACS70'])
sed['PACS70_err'][pacs70_chng] = sed['PACS70'][pacs70_chng]
sed['PACS70'][pacs70_chng] = np.nan

# Upload info on BAT AGN for redshift and luminosity distance
bat_info = pd.read_csv('../../bat-data/bat_info.csv', index_col=0)

filt_use = np.array(['PACS70', 'PACS160', 'PSW', 'PMW', 'PLW'])
filt_err = np.array([s+'_err' for s in filt_use])
waves = np.array([70., 160., 250., 350., 500.])

# Uncomment to fit sources with detections at all wavelengths
# sed_use = sed.dropna(how='any')

# Uncomment to fit sources with only N undetected points.
# Change the integer on the right side of '==' to N.
sed_use = sed[np.sum(np.isfinite(sed[filt_use].values), axis=1) >= 3]

names_use = sed_use.index
#names_use = ['ESO103-035']

base_model = bat_model.Greybody(0.0, 25., 2.0)
lev_marq = apy_fit.LevMarLSQFitter()
bayes = bat_fit.SEDBayesFitter(threads=8)

# Fix parameters
base_model.beta.fixed = True

for n in names_use:
#for n in names_use[0:1]:
    print 'Fitting: ', n
    src_sed = sed_use.loc[n][filt_use]
    flux = np.array(src_sed)
    src_err = sed_use.loc[n][filt_err]
    flux_err = np.array(src_err)
	
    src_z = bat_info.loc[n]['Redshift']
    src_lumD = bat_info.loc[n]['Dist_[Mpc]']

    model_ml = base_model.copy()
    model_ml.set_redshift(src_z)
    model_ml.set_lumD(src_lumD)
    
    if not np.isnan(src_sed['PSW']):
        mdust_init = np.log10(src_sed['PSW']/model_ml(250))
    else:
        mdust_init = np.log10(src_sed['PSW_err']/model_ml(250))

    model_ml.mdust.value = mdust_init

    model_ml.beta.fixed = True
    
    detected = np.isfinite(flux)
    model_init = lev_marq(model_ml, waves[detected], flux[detected], weights=1/flux_err[detected],
                          maxiter=1000)
    
    model_final = bayes.fit(model_init, waves, flux, yerr=flux_err, use_nondetect=True,
                            filts=filt_use)

    print 'Plotting the fit: ', n
    fig_fit = bat_plot.plot_fit(waves, flux, model_final, obs_err=flux_err,
                                plot_components=False, plot_mono_fluxes=True,
                                filts=filt_use, plot_fit_spread=True,
                                name=n, plot_params=True)

    fig_fit.savefig('single_mbb_bayes_results/beta_fixed_2/sed_plots/' + n +
                    '_single_mbb_bayes_beta2_sed_fit.png',
                    bbox_inches='tight')
    plt.close(fig_fit)
    fig_triangle = bat_plot.plot_triangle(model_final)
    fig_triangle.savefig('single_mbb_bayes_results/beta_fixed_2/triangle_plots/' + n +
                         '_single_mbb_bayes_beta2_triangle.png',
                         bbox_inches='tight')
    plt.close(fig_triangle)
    print 'Saving the fit: ', n
    fit_dict = {'name': n,
                'flux': flux,
                'flux_err': flux_err,
                'best_fit_model': model_final,
                'filters': filt_use,
                'waves': waves}
    pickle_file = open('single_mbb_bayes_results/beta_fixed_2/pickles/' + n +
                       '_single_mbb_bayes_beta2.pickle', 'wb')
    pickle.dump(fit_dict, pickle_file)
    pickle_file.close()
