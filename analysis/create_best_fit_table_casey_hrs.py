from glob import glob
import sys
import numpy as np
import pandas as pd
import pickle

dir_sed = '/Users/ttshimiz/Github/bat-agn-sed-fitting/'
sys.path.append(dir_sed)

files = glob(dir_sed+'analysis/casey_bayes_results/hrs_beta_fixed_2_wturn_gaussianPrior/pickles/*')

names = np.array([x.split('/')[-1].split('_')[0] for x in files])

nsrc = len(files)
tdust = np.zeros(nsrc)
tdust_err_up = np.zeros(nsrc)
tdust_err_down = np.zeros(nsrc)
mdust = np.zeros(nsrc)
mdust_err_up = np.zeros(nsrc)
mdust_err_down = np.zeros(nsrc)
beta = np.zeros(nsrc)
beta_err_up = np.zeros(nsrc)
beta_err_down = np.zeros(nsrc)
alpha = np.zeros(nsrc)
alpha_err_up = np.zeros(nsrc)
alpha_err_down = np.zeros(nsrc)
norm_pow = np.zeros(nsrc)
norm_pow_err_up = np.zeros(nsrc)
norm_pow_err_down = np.zeros(nsrc)
wturn = np.zeros(nsrc)
wturn_err_up = np.zeros(nsrc)
wturn_err_down = np.zeros(nsrc)

lir_total = np.zeros(nsrc)
lir_total_err_up = np.zeros(nsrc)
lir_total_err_down = np.zeros(nsrc)
lir_bb = np.zeros(nsrc)
lir_bb_err_up = np.zeros(nsrc)
lir_bb_err_down = np.zeros(nsrc)
lir_powlaw = np.zeros(nsrc)
lir_powlaw_err_up = np.zeros(nsrc)
lir_powlaw_err_down = np.zeros(nsrc)
#agn_frac = np.zeros(nsrc)
#agn_frac_err_up = np.zeros(nsrc)
#agn_frac_err_down = np.zeros(nsrc)

for i in range(nsrc):

    f = open(files[i], 'rb')
    fit_result = pickle.load(f)
    mod = fit_result['best_fit_model']
    f.close()
    
    tdust[i] = mod.tdust.value
    mdust[i] = mod.mdust.value
    beta[i] = mod.beta.value
    alpha[i] = mod.alpha.value
    norm_pow[i] = mod.pownorm.value
    wturn[i] = mod.wturn.value
    
    tdust_err_up[i] = mod.param_errs[np.array(mod.param_names) == 'tdust', 1]
    tdust_err_down[i] = mod.param_errs[np.array(mod.param_names) == 'tdust', 0]
    mdust_err_up[i] = mod.param_errs[np.array(mod.param_names) == 'mdust', 1]
    mdust_err_down[i] = mod.param_errs[np.array(mod.param_names) == 'mdust', 0]
    alpha_err_up[i] = mod.param_errs[np.array(mod.param_names) == 'alpha', 1]
    alpha_err_down[i] = mod.param_errs[np.array(mod.param_names) == 'alpha', 0]
    beta_err_up[i] = mod.param_errs[np.array(mod.param_names) == 'beta', 1]
    beta_err_down[i] = mod.param_errs[np.array(mod.param_names) == 'beta', 1]
    wturn_err_up[i] = mod.param_errs[np.array(mod.param_names) == 'wturn', 1]
    wturn_err_down[i] = mod.param_errs[np.array(mod.param_names) == 'wturn', 0]
    norm_pow_err_up[i] = mod.param_errs[np.array(mod.param_names) == 'pownorm', 1]
    norm_pow_err_down[i] = mod.param_errs[np.array(mod.param_names) == 'pownorm', 0]
    
    lir_total[i] = np.log10(mod.calc_luminosity())
    lir_bb[i] = np.log10(mod.calc_bb_lum())
    lir_powlaw[i] = np.log10(mod.calc_plaw_lum()) 
    #agn_frac[i] = (10**lir_powlaw[i] - 1./3.*10**lir_bb[i])/10**lir_total[i]
    
    lir_total_sample = np.zeros(1000)
    lir_bb_sample = np.zeros(1000)
    lir_powlaw_sample = np.zeros(1000)
    #agn_frac_sample = np.zeros(1000)
    
    chain = mod.chain_nb
    dummy = mod.copy()
    for j in range(1000):
        dummy.mdust.value = chain[j,0]
        dummy.tdust.value = chain[j, 1]
        dummy.pownorm.value = chain[j, 2]
        dummy.alpha.value = chain[j, 3]
        dummy.wturn.value = chain[j, 4]
        lir_total_sample[j] = np.log10(dummy.calc_luminosity())
        lir_bb_sample[j] = np.log10(dummy.calc_bb_lum())
        lir_powlaw_sample[j] = np.log10(dummy.calc_plaw_lum())
        #agn_frac_sample[j] = (10**lir_powlaw_sample[j] - 1./3.*10**lir_bb_sample[j])/10**lir_total_sample[j]
        
    lir_total_err_down[i] = np.percentile(lir_total_sample, 16)
    lir_total_err_up[i] = np.percentile(lir_total_sample, 84)
    lir_bb_err_down[i] = np.percentile(lir_bb_sample, 16)
    lir_bb_err_up[i] = np.percentile(lir_bb_sample, 84)
    lir_powlaw_err_down[i] = np.percentile(lir_powlaw_sample, 16)
    lir_powlaw_err_up[i] = np.percentile(lir_powlaw_sample, 84)    
    #agn_frac_err_down[i] = np.percentile(agn_frac_sample, 16)
    #agn_frac_err_up[i] = np.percentile(agn_frac_sample, 84)
    
df = pd.DataFrame({'mdust':pd.Series(mdust, index=names),
                   'mdust_16':pd.Series(mdust_err_down, index=names),
                   'mdust_84':pd.Series(mdust_err_up, index=names),
                   'tdust':pd.Series(tdust, index=names),
                   'tdust_16':pd.Series(tdust_err_down, index=names),
                   'tdust_84':pd.Series(tdust_err_up, index=names),
                   'norm_pow':pd.Series(norm_pow, index=names),
                   'norm_pow_16':pd.Series(norm_pow_err_down, index=names),
                   'norm_pow_84':pd.Series(norm_pow_err_up, index=names),
                   'alpha':pd.Series(alpha, index=names),
                   'alpha_16':pd.Series(alpha_err_down, index=names),
                   'alpha_84':pd.Series(alpha_err_up, index=names),
                   'wturn':pd.Series(wturn, index=names),
                   'wturn_16':pd.Series(wturn_err_down, index=names),
                   'wturn_84':pd.Series(wturn_err_up, index=names),
                   'lir_total':pd.Series(lir_total, index=names),
                   'lir_total_16':pd.Series(lir_total_err_down, index=names),
                   'lir_total_84':pd.Series(lir_total_err_up, index=names),
                   'lir_bb':pd.Series(lir_bb, index=names),
                   'lir_bb_16':pd.Series(lir_bb_err_down, index=names),
                   'lir_bb_84':pd.Series(lir_bb_err_up, index=names),
                   'lir_powlaw':pd.Series(lir_powlaw, index=names),
                   'lir_powlaw_16':pd.Series(lir_powlaw_err_down, index=names),
                   'lir_powlaw_84':pd.Series(lir_powlaw_err_up, index=names)})
                   #'agn_frac':pd.Series(agn_frac, index=names),
                   #'agn_frac_16':pd.Series(agn_frac_err_down, index=names),
                   #'agn_frac_84':pd.Series(agn_frac_err_up, index=names)})

df.to_csv(dir_sed+'analysis/casey_bayes_results/hrs_beta_fixed_2_wturn_gaussianPrior/final_fit_results_beta_fixed_2_wturn_gaussianPrior_hrs.csv',
          index_label='Name')