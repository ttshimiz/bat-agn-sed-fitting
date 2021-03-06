# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:18:03 2015

@author: ttshimiz
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from fitting import Filters
import seaborn
seaborn.set_style('ticks')

tex_names = {'mdust': r'$M_{dust}$',
             'tdust': r'$T_{dust}$',
             'alpha': r'$\alpha$',
             'beta': r'$\beta$',
             'pownorm': r'$N_{pl}$',
             'wturn': r'$\lambda_{turn}$',
             'norm': r'$N_{old}$',
             'temp': r'$T_{old}$'}


def plot_fit(waves, obs_flux, model, model_waves=np.arange(1, 1000),
             obs_err=None, plot_components=False, comp_colors=None,
             plot_mono_fluxes=False, filts=None,
             plot_fit_spread=False, nspread=1000,
             name=None, plot_params=False,
             seaborn_context='notebook', fig=None, subplot=None):

    
    seaborn.set(context=seaborn_context)
    seaborn.set(context='notebook', color_codes=True, style='ticks')
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['mathtext.fontset'] = 'stixsans'
    red = seaborn.xkcd_rgb['pale red']
    blue = seaborn.xkcd_rgb['denim blue']
    lt_blue = seaborn.xkcd_rgb['pastel blue']
    
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        if subplot is None:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(subplot)
    
    zcorr = 1 + model.redshift
    #median_model = model(model_waves/zcorr) * zcorr

    #ax.loglog(model_waves, median_model, color=blue, label='Best Fit Model')

    if plot_fit_spread:
        dummy = model.copy()
        fixed = np.array([dummy.fixed[n] for n in dummy.param_names])
        param_rand = np.random.randint(low=0, high=len(model.chain_nb),
                                       size=nspread)
        rand_sed = np.zeros((nspread, len(model_waves)))

        for i in range(nspread):

            dummy.parameters[~fixed] = model.chain_nb[param_rand[i]]
            rand_sed[i, :] = dummy(model_waves/zcorr) * zcorr

        model_2_5, model_50, model_97_5 = np.percentile(rand_sed, [2.5, 50.0, 97.5], axis=0)
        ax.loglog(model_waves, model_50, color=blue, label='Best Fit Model')
        ax.fill_between(model_waves, model_2_5, model_97_5, color=lt_blue,
                        alpha=0.5, label='_nolabel')

    if plot_components:
        ncomps = model.n_components
        if comp_colors is None:
            comp_colors = seaborn.color_palette('colorblind',
                                                n_colors=ncomps+1)[1:]
        comps = model.eval_comps(model_waves/zcorr) * zcorr
        for i in range(ncomps):
            ax.loglog(model_waves, comps[i, :], ls='--',
                      label=model.comp_names[i], color=comp_colors[i])

    undetected = np.isnan(obs_flux)
    if plot_mono_fluxes:
        dummy2 = model.copy()
        filters = Filters()
        fwaves = filters.filter_waves
        filts = np.array(filts)[~undetected]
        mono_fluxes = np.array([filters.calc_mono_flux(f, fwaves[f],
                                dummy2(fwaves[f]/zcorr))*zcorr for f in filts])
        ax.plot(waves[~undetected], mono_fluxes, marker='s', ls='None',
                color=red, label='Model Fluxes')
        if sum(undetected) > 0:
            fluxes = np.hstack([mono_fluxes, obs_flux[~undetected],
                                obs_err[undetected]])
        else:
            fluxes = np.hstack([mono_fluxes, obs_flux])
    else:
        if sum(undetected) > 0:
            fluxes = np.hstack([obs_flux[~undetected],
                                obs_err[undetected]])
        else:
            fluxes = obs_flux

    if obs_err is None:
        ax.plot(waves[~undetected], obs_flux[~undetected], marker='o',
                ls='None', color='k', label='Observed Fluxes')
    else:
        ax.errorbar(waves[~undetected], obs_flux[~undetected],
                    yerr=obs_err[~undetected], marker='o', ls='None',
                    color='k', label='Observed Fluxes')
        if sum(undetected) > 0:
            ax.quiver(waves[undetected], obs_err[undetected],
                      np.zeros(sum(undetected)),
                      -np.ones(sum(undetected)),
                      width=0.005, color='r', scale=15, scale_units='height',
                      units='width')
            #ax.plot(waves[undetected], obs_err[undetected],
            #        marker='o', mec=red, mfc='None', mew=1.5, ls='None')
    if plot_params:
        fs = mpl.rcParams['legend.fontsize']
        for i, n in enumerate(model.param_names):
            ln = tex_names[n]
            p = model.parameters[i]
            l = p - model.param_errs[i, 0]
            u = model.param_errs[i, 1] - p
            s = r'%s$ = %.2f^{+%.2f}_{-%.2f}$' % (ln, p, u, l)

            ax.text(0.97, 0.97-i*0.06, s, ha='right', va='top',
                    transform=ax.transAxes, fontsize=fs)

    if name is not None:
        ax.set_title(name, fontsize=14)

    ax.set_xlim([1, 1000])
    ax.set_ylim([10**(-0.5)*min(fluxes), 10**(0.5)*max(fluxes)])
    #ax.legend(loc='upper left')
    ax.set_xlabel('Wavelength [micron]')
    ax.set_ylabel('Flux Density [Jy]')
    seaborn.despine()
    return fig


def plot_triangle(model, quantiles=[0.16, 0.5, 0.84]):

    import triangle
    seaborn.set_style('ticks')
    fixed  = np.array([model.fixed[n] for n in model.param_names])
    labels = [tex_names[n] for n in np.array(model.param_names)[~fixed]]

    fig = triangle.corner(model.chain_nb, quantiles=quantiles,
                          labels=labels, verbose=False, plot_contours=False)
    seaborn.despine()
    return fig


def plot_fit_dale14(waves, obs_flux, model, obs_err=None,
                    plot_mono_fluxes=False, filts=None,
                    name=None, plot_params=False,
                    seaborn_context='notebook'):

    seaborn.set(context=seaborn_context)
    red = seaborn.xkcd_rgb['pale red']
    blue = seaborn.xkcd_rgb['denim blue']
    lt_blue = seaborn.xkcd_rgb['pastel blue']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    model_best_fit = model.best_fit['sed']
    model_waves = model.waves
    
    ax.loglog(model_waves*(1+model.redshift), model_best_fit, color=blue,
              label='Best Fit Model')
    
    undetected = np.isnan(obs_flux)
    if plot_mono_fluxes:
        filters = Filters()
        fwaves = filters.filter_waves
        filts = np.array(filts)[~undetected]
        zcorr = 1 + model.redshift
        mono_fluxes = np.array([filters.calc_mono_flux(f, model_waves*(1+model.redshift),
                                model_best_fit) for f in filts])
        ax.plot(waves[~undetected], mono_fluxes, marker='^', ls='None',
                color=red, label='Model Fluxes')
        if sum(undetected) > 0:
            fluxes = np.hstack([mono_fluxes, obs_flux[~undetected],
                                obs_err[undetected]])
        else:
            fluxes = np.hstack([mono_fluxes, obs_flux])
    else:
        if sum(undetected) > 0:
            fluxes = np.hstack([obs_flux[~undetected],
                                obs_err[undetected]])
        else:
            fluxes = obs_flux

    if obs_err is None:
        ax.plot(waves[~undetected], obs_flux[~undetected], marker='o',
                ls='None', color='k', label='Observed Fluxes')
    else:
        ax.errorbar(waves[~undetected], obs_flux[~undetected],
                    yerr=obs_err[~undetected], marker='o', ls='None',
                    color='k', label='Observed Fluxes')
        if sum(undetected) > 0:
            ax.quiver(waves[undetected], obs_err[undetected],
                      np.zeros(sum(undetected)),
                      -0.025*np.ones(sum(undetected)),
                      width=0.004, color='k')
    
    if plot_params:
        fs = mpl.rcParams['legend.fontsize']
        s_alpha = r'%s$ = %.2f$' % (tex_names['alpha'], model.best_fit['alpha'])
        s_fracAGN = r'%s$ = %.2f$' % ('fracAGN', model.best_fit['fracAGN'])
        ax.text(0.97, 0.97, s_alpha, ha='right', va='top',
                    transform=ax.transAxes, fontsize=fs)        
        ax.text(0.97, 0.97-0.06, s_fracAGN, ha='right', va='top',
                    transform=ax.transAxes, fontsize=fs)

    if name is not None:
        ax.set_title(name)

    ax.set_xlim([1, 1000])
    ax.set_ylim([10**(-0.5)*min(fluxes), 10**(0.5)*max(fluxes)])
    ax.legend(loc='upper left')
    ax.set_xlabel('Wavelength [micron]')
    ax.set_ylabel('Flux Density [Jy]')

    return fig

def plot_fit_decompir(waves, obs_flux, model, obs_err=None,
                      plot_mono_fluxes=False, filts=None,
                      plot_components=False, comp_colors=None,
                      name=None, plot_params=False,
                      seaborn_context='notebook'):

    seaborn.set(context=seaborn_context)
    red = seaborn.xkcd_rgb['pale red']
    blue = seaborn.xkcd_rgb['denim blue']
    lt_blue = seaborn.xkcd_rgb['pastel blue']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    model_best_fit = model.best_fit['sed']
    model_waves = model.waves
    
    ax.loglog(model_waves*(1+model.redshift), model_best_fit, color=blue,
              label='Best Fit Model')
    
    if plot_components:
        ncomps = 2
        if comp_colors is None:
            comp_colors = seaborn.color_palette('colorblind',
                                                n_colors=ncomps+1)[1:]
        ax.loglog(model_waves*(1+model.redshift), model.best_fit['agn_sed'], ls='--',
                  label='AGN', color=comp_colors[1])
        ax.loglog(model_waves*(1+model.redshift), model.best_fit['host_sed'], ls='--',
                  label='Host', color=comp_colors[0])
    
    undetected = np.isnan(obs_flux)
    if plot_mono_fluxes:
        filters = Filters()
        fwaves = filters.filter_waves
        filts = np.array(filts)[~undetected]
        zcorr = 1 + model.redshift
        mono_fluxes = np.array([filters.calc_mono_flux(f, model_waves*(1+model.redshift),
                                model_best_fit) for f in filts])
        ax.plot(waves[~undetected], mono_fluxes, marker='^', ls='None',
                color=red, label='Model Fluxes')
        if sum(undetected) > 0:
            fluxes = np.hstack([mono_fluxes, obs_flux[~undetected],
                                obs_err[undetected]])
        else:
            fluxes = np.hstack([mono_fluxes, obs_flux])
    else:
        if sum(undetected) > 0:
            fluxes = np.hstack([obs_flux[~undetected],
                                obs_err[undetected]])
        else:
            fluxes = obs_flux

    if obs_err is None:
        ax.plot(waves[~undetected], obs_flux[~undetected], marker='o',
                ls='None', color='k', label='Observed Fluxes')
    else:
        ax.errorbar(waves[~undetected], obs_flux[~undetected],
                    yerr=obs_err[~undetected], marker='o', ls='None',
                    color='k', label='Observed Fluxes')
        if sum(undetected) > 0:
            ax.quiver(waves[undetected], obs_err[undetected],
                      np.zeros(sum(undetected)),
                      -0.025*np.ones(sum(undetected)),
                      width=0.004, color='k')
    
    if plot_params:
        fs = mpl.rcParams['legend.fontsize']
        s_alpha = r'%s = %s' % ('Host', model.best_fit['host_name'])
        s_agn_sed = r'%s = %s' % ('AGN', model.best_fit['agn_name'])
        s_fracAGN = r'%s$ = %.2f$' % ('fracAGN', model.best_fit['fracAGN'])
        ax.text(0.97, 0.97, s_alpha, ha='right', va='top',
                    transform=ax.transAxes, fontsize=fs)        
        ax.text(0.97, 0.97-0.06, s_agn_sed, ha='right', va='top',
                    transform=ax.transAxes, fontsize=fs)
        ax.text(0.97, 0.97-2*0.06, s_fracAGN, ha='right', va='top',
                    transform=ax.transAxes, fontsize=fs)

    if name is not None:
        ax.set_title(name)

    ax.set_xlim([1, 1000])
    ax.set_ylim([10**(-0.5)*min(fluxes), 10**(0.5)*max(fluxes)])
    ax.legend(loc='upper left')
    ax.set_xlabel('Wavelength [micron]')
    ax.set_ylabel('Flux Density [Jy]')

    return fig