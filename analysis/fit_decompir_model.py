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

# Uncomment to fit sources with only N undetected points.
# Change the integer on the right side of '==' to N.
sed_use = sed[np.sum(np.isfinite(sed[filt_use].values), axis=1) >= 4]

names_use = sed_use.index
#names_use = ['CGCG493-002']

# Base Decompir 2014 model
decompir_model = bat_model.DecompIR()

for n in names_use:
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

    decompir_model.set_redshift(src_z)
    decompir_model.set_lumD(src_lumD)

    decompir_model.fit(waves_use, flux_use, yerr=flux_err_use, filts=filt_detected)
    print decompir_model.best_fit['host_name']
    print 'Plotting the fit: ', n
    fig = bat_plot.plot_fit_decompir(waves, flux, decompir_model, obs_err=flux_err,
                                   plot_mono_fluxes=True, filts=filt_use,
                                   name=n, plot_params=True, plot_components=True)
    fig.savefig('./decompir_results/sb+arp220/sed_plots/'+n+'_decompir_best_fit_sed.png',
                bbox_inches='tight')
    plt.close(fig)

    print 'Saving the fit: ', n
    fit_dict = {'name': n,
                'flux': flux,
                'flux_err': flux_err,
                'best_fit_model': decompir_model,
                'filters': filt_use,
                'waves': waves}
    pickle_file = open('./decompir_results/sb+arp220/pickles/'+n+'_decompir_best_fit_model.pickle',
                       'wb')
    pickle.dump(fit_dict, pickle_file)
    pickle_file.close()
