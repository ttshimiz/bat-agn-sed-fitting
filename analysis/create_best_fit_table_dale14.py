from glob import glob
import sys
import numpy as np
import pandas as pd
import pickle

dir_sed = '/Users/ttshimiz/Github/bat-agn-sed-fitting/'
sys.path.append(dir_sed)

files = glob(dir_sed+'analysis/dale14_results/pickles/*')
frac_agn_totals = pd.read_table(dir_sed+'Dale14/fraction_AGN_bolometric.dat', skiprows=1, index_col=0, delim_whitespace=True)
names = np.array([x.split('/')[-1].split('_')[0] for x in files])

nsrc = len(files)
alpha = np.zeros(nsrc)
frac_agn_mir = np.zeros(nsrc)
frac_agn_total = np.zeros(nsrc)
chi2 = np.zeros(nsrc)
lir_total = np.zeros(nsrc)
norm = np.zeros(nsrc)

for i in range(nsrc):

    f = open(files[i], 'rb')
    fit_result = pickle.load(f)
    mod = fit_result['best_fit_model']
    f.close()
    
    alpha[i] = mod.best_fit['alpha']
    frac_agn_mir[i] = mod.best_fit['fracAGN']
    chi2[i] = mod.best_fit['chi2']
    lir_total[i] = mod.best_fit['LIR']
    norm[i] = mod.best_fit['norm']
    
    if frac_agn_mir[i] == 0:
        fa = '00%'
    elif frac_agn_mir[i] == 0.05:
        fa = '05%'
    else:
        fa = str(int(frac_agn_mir[i]*100))+'%'
    
    frac_agn_total[i] = frac_agn_totals.loc[alpha[i], fa]/100.

df = pd.DataFrame({'alpha': pd.Series(alpha, index=names),
                   'agn_frac_mir': pd.Series(frac_agn_mir, index=names),
                   'agn_frac': pd.Series(frac_agn_total, index=names),
                   'chi2': pd.Series(chi2, index=names),
                   'lir_total': pd.Series(lir_total, index=names),
                   'norm': pd.Series(norm, index=names)})
df.to_csv(dir_sed+'analysis/dale14_results/final_fit_results_dale14.csv',
          index_label='Name')
