#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
import pickle

_OUTFILE = 'out/pca/pca.pkl'

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

def run():
    pca = pickle.load(open(_OUTFILE, 'rb'))
    plot_expl_var_ratio(pca)
    for i in range(50):
        plot_component(pca, i)

if __name__ == '__main__':
    run()
