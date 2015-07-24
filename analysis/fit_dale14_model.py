# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:36:40 2015

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

# Upload info on BAT AGN for redshift and luminosity distance
bat_info = pd.read_csv('../../bat-data/bat_info.csv', index_col=0)

filt_use = np.array(['W3', 'W4', 'PACS70', 'PACS160', 'PSW', 'PMW', 'PLW'])
filt_err = np.array([s+'_err' for s in filt_use])
waves = np.array([12., 22., 70., 160., 250., 350., 500.])

# Uncomment to fit sources with only N undetected points.
# Change the integer on the right side of '==' to N.
sed_use = sed[np.sum(np.isnan(sed.values), axis=1) <= 4]

names_use = sed_use.index

# Base Dale 2014 model
dale_model = bat_model.Dale2014()

for n in names_use:
    print 'Fitting: ', n
    src_sed = sed_use.loc[n][filt_use]
    flux = np.array(src_sed)
    flux_detected = np.isfinite(flux)
    flux_use = flux[flux_detected]
    src_err = sed_use.loc[n][filt_err]
    flux_err = np.array(src_err)
    flux_err_use = flux_err[flux_detected]
    filt_detected = filt_use[flux_detected]
    waves_use = waves[flux_detected]
	
    src_z = bat_info.loc[n]['Redshift']
    src_lumD = bat_info.loc[n]['Dist_[Mpc]']

    dale_model.set_redshift(src_z)
    dale_model.set_lumD(src_lumD)

    dale_model.fit(waves_use, flux_use, yerr=flux_err_use, filts=filt_detected)
    print 'Plotting the fit: ', n
    fig = bat_plot.plot_fit_dale14(waves, flux, dale_model, obs_err=flux_err,
                                   plot_mono_fluxes=True, filts=filt_use,
                                   name=n, plot_params=True)
    fig.savefig('./dale14_results/sed_plots/'+n+'_dale14_best_fit_sed.png',
                bbox_inches='tight')
    plt.close(fig)

    print 'Saving the fit: ', n
    fit_dict = {'name': n,
                'flux': flux,
                'flux_err': flux_err,
                'best_fit_model': dale_model,
                'filters': filt_use,
                'waves': waves}
    pickle_file = open('./dale14_results/pickles/'+n+'_dale14_best_fit_model.pickle',
                       'wb')
    pickle.dump(fit_dict, pickle_file)
    pickle_file.close()
    

