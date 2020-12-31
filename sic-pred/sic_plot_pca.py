#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pandas as pd
import pickle

_OUTFILE = 'out/pca/pca.pkl'

def plot_errors(df_all, figname):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
    for npc, df in df_all.items():
        for ax, stat, units in zip(axes.flatten(),
                ['Bias', 'RMSE', 'Extent_Bias', 'IIEE'],
                ['', '', r', $10^6$km$^2$', r', $10^6$km$^2$'],
                ):
            ax.plot(df.Date, df[stat], label='N=%i' %npc)
            ax.set_ylabel(stat.replace('_', ' ') + units)
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0,0].legend()
    fig.autofmt_xdate()
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print(f'Saving {figname}')
    fig.savefig(figname)
    plt.close()

def plot_component(pca, i):
    pc = pca.map_to_grid(pca.get_component(i))
    evr = pca.pca.explained_variance_ratio_[i]
    figname = f'out/pca/pc{i}.png'
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_title(f'Principal component {i}: explained variance = {100*evr}%')
    im = ax.imshow(pc)
    fig.colorbar(im)
    print(f"Saving {figname}")
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    fig.savefig(figname)
    plt.close()

def plot_expl_var_ratio(pca):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.cumsum(pca.pca.explained_variance_ratio_))
    figname = 'out/pca/explained_variance_ratio.png'
    print(f'Saving {figname}')
    fig.savefig(figname)

def plot_components(pca):
    for i in range(50):
        plot_component(pca, i)

def do_eval(pca):
    df_all = dict()
    for npc in [1,10,20,50,100]:
        outfile = f'out/pca/pca{npc}.csv'
        if os.path.exists(outfile):
            print(f'Reading {outfile}')
            df = pd.read_csv(outfile)
        else:
            start = pca.datetimes[0]
            end = pca.datetimes[-1]
            df = pca.comp_all_errors(start, end, n_components=npc)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            print(f'Saving {outfile}')
            df.set_index('Date').to_csv(outfile)
        df_all[npc] = df
    plot_errors(df_all, 'out/pca/errors.png')

def run():
    pca = pickle.load(open(_OUTFILE, 'rb'))
    plot_expl_var_ratio(pca)
    plot_components(pca)
    do_eval(pca)

if __name__ == '__main__':
    run()
