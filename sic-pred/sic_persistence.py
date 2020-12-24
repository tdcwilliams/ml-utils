#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from netCDF4 import Dataset

from sic_pred_base import SicPredPersistence

_OSISAF_DIR = os.path.join(
        os.getenv('CLUSTER_ROOT_DIR'), 'data/OSISAF_ice_conc/polstere')
_NAME_MASK = os.path.join(_OSISAF_DIR,
        '%Y_nh_polstere/ice_conc_nh_polstere-100_multi_%Y%m%d1200.nc')
_START = dt.datetime(2019,1,1)
_END = dt.datetime(2019,12,31)
_MAX_LAG = 7

def make_plots(df_all, figname):
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for lag, df in df_all.items():
        ax1.plot(df.Date, df.Bias, label='Lag=%i' %lag)
        ax2.plot(df.Date, df.RMSE, label='Lag=%i' %lag)
    ax1.set_title('Bias')
    ax2.set_title('RMSE')
    for ax in [ax1, ax2]:
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.legend()
    fig.autofmt_xdate()
    print(f'Saving {figname}')
    fig.savefig(figname)
    plt.close()

def run():
    df_all = dict()
    for lag in range(1, _MAX_LAG + 1):
        fc = SicPredPersistence(lag)
        df = fc.comp_all_errors(_START, _END)
        outfile = f'out/persistence-lag{lag}.csv'
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        print(f'Saving {outfile}')
        df.set_index('Date').to_csv(outfile)
        df_all[lag] = df
    make_plots(df_all, 'out/persistence.png')

if __name__ == '__main__':
    run()
