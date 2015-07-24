
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

# Uncomment to fit sources with detections at all wavelengths
# sed_use = sed.dropna(how='any')

# Uncomment to fit sources with only N undetected points.
# Change the integer on the right side of '==' to N.
sed_use = sed[np.sum(np.isnan(sed.values), axis=1) == 1]

names_use = sed_use.index

base_model = bat_model.GreybodyPowerlaw(0.0, 25., 1.8, 0.0, 2.0, 50.0)
lev_marq = apy_fit.LevMarLSQFitter()
bayes = bat_fit.SEDBayesFitter(threads=8)

# Fix parameters
base_model.wturn.fixed = True
base_model.beta.fixed = True

for n in names_use:

    pfile = open('casey_bayes_results/wturn_fixed50/pickles/' + n +
                 '_casey_bayes_wturn_fixed50.pickle', 'rb')

    data = pickle.load(pfile)

    model = data['best_fit_model']
    fig_triangle = bat_plot.plot_triangle(model)
    fig_triangle.savefig('casey_bayes_results/wturn_fixed50/triangle_plots/' + n + '_casey_bayes_wturn_fixed50_triangle.png', bbox_inches='tight')
    plt.close(fig_triangle)
