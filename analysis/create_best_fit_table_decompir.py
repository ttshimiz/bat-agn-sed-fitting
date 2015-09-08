from glob import glob
import sys
import numpy as np
import pandas as pd
import pickle

dir_sed = '/Users/ttshimiz/Github/bat-agn-sed-fitting/'
sys.path.append(dir_sed)

files = glob(dir_sed+'analysis/decompir_results/sb_only/pickles/*')

names = np.array([x.split('/')[-1].split('_')[0] for x in files])

nsrc = len(files)
host_name = np.zeros(nsrc, dtype='|S7')
agn_name = np.zeros(nsrc, dtype='|S7')
frac_agn = np.zeros(nsrc)
chi2 = np.zeros(nsrc)
lir_total = np.zeros(nsrc)
agn_norm = np.zeros(nsrc)
host_norm = np.zeros(nsrc)

for i in range(nsrc):

    f = open(files[i], 'rb')
    fit_result = pickle.load(f)
    mod = fit_result['best_fit_model']
    f.close()
    
    host_name[i] = mod.best_fit['host_name']
    agn_name[i] = mod.best_fit['agn_name']
    frac_agn[i] = mod.best_fit['fracAGN']
    chi2[i] = mod.best_fit['chi2']
    lir_total[i] = mod.best_fit['LIR']
    agn_norm[i] = mod.best_fit['norms'][0]
    host_norm[i] = mod.best_fit['norms'][1]
    
df = pd.DataFrame({'host_name': pd.Series(host_name, index=names),
                   'agn_name': pd.Series(agn_name, index=names),
                   'agn_frac': pd.Series(frac_agn, index=names),
                   'chi2': pd.Series(chi2, index=names),
                   'lir_total': pd.Series(lir_total, index=names),
                   'agn_norm': pd.Series(agn_norm, index=names),
                   'host_norm': pd.Series(host_norm, index=names)})
df.to_csv(dir_sed+'analysis/decompir_results/sb_only/final_fit_results_decompir_sb_and_arp220.csv',
          index_label='Name')
