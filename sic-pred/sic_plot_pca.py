#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pandas as pd
import pickle

_OUTFILE = 'out/pca/pca.pkl'

def plot_component(sic_pca, i):
    pc = sic_pca.map_to_grid(sic_pca.get_component(i))
    evr = sic_pca.pca.explained_variance_ratio_[i]
    figname = f'out/pca/components/pc{i}.png'
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_title(f'Principal component {i}: explained variance = {100*evr}%')
    im = ax.imshow(pc)
    fig.colorbar(im)
    print(f"Saving {figname}")
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    fig.savefig(figname)
    plt.close()

def plot_expl_var_ratio(sic_pca):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.cumsum(sic_pca.pca.explained_variance_ratio_))
    figname = 'out/pca/explained_variance_ratio.png'
    print(f'Saving {figname}')
    fig.savefig(figname)

def plot_components(sic_pca):
    for i in range(10):
        plot_component(sic_pca, i)

def do_eval(sic_pca):
    pfile = 'out/persistence-lag1.csv'
    print(f'Reading {pfile}')
    df_all = dict(persistence=pd.read_csv(pfile))
    for npc in [1, 5, 10, 20, 50, 100, 356]:
        outfile = f'out/pca/pca{npc}.csv'
        if not os.path.exists(outfile):
            start = sic_pca.datetimes[0]
            end = sic_pca.datetimes[-1]
            df = sic_pca.comp_all_errors(start, end, n_components=npc,
                    figmask=f'out/pca/maps{npc}/error-maps-%Y%m%d.png')
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            print(f'Saving {outfile}')
            df.set_index('Date').to_csv(outfile)
        print(f'Reading {outfile}')
        df = pd.read_csv(outfile)
        df_all[f'N={npc}'] = df
    sic_pca.plot_errors(df_all, 'out/pca/errors.png')

def run():
    sic_pca = pickle.load(open(_OUTFILE, 'rb'))
    plot_expl_var_ratio(sic_pca)
    plot_components(sic_pca)
    do_eval(sic_pca)

if __name__ == '__main__':
    run()
