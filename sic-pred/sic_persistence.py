#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from matplotlib import pyplot as plt

from netCDF4 import Dataset

_OSISAF_DIR = os.path.join(
        os.getenv('CLUSTER_ROOT_DIR'), 'data/OSISAF_ice_conc/polstere')
_NAME_MASK = os.path.join(_OSISAF_DIR,
        '%Y_nh_polstere/ice_conc_nh_polstere-100_multi_%Y%m%d1200.nc')

def get_conc(dto):
    #print(f'Reading SIC from {f}')
    f = dto.strftime(_NAME_MASK)
    with Dataset(f, 'r') as ds:
        return .01*ds.variables['ice_conc'][0].filled(np.nan)

def comp_errors(sic, sic_hat):
    errors = dict()
    errors['Bias'] = np.nanmean(sic_hat-sic)
    errors['RMSE'] = np.sqrt(np.nanmean((sic_hat-sic)**2))
    return errors

def comp_all_errors():
    start = dt.datetime(2018,8,1)
    #end = dt.datetime(2018,8,14)
    end = dt.datetime(2019,7,31)
    days = 1 + (end - start).days
    dates = [start + dt.timedelta(i) for i in range(days)]
    lags = [i+1 for i in range(7)]
    errors = defaultdict(list)
    for dto in dates:
        print(dto)
        sic = get_conc(dto)
        for lag in lags:
            errors['Date'] += [dto]
            errors['Lag'] += [lag]
            sic_hat = get_conc(dto - dt.timedelta(lag))
            for k, v in comp_errors(sic, sic_hat).items():
                errors[k] += [v]
    return pd.DataFrame(errors)

def make_plots(df, figname):
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for lag in df.Lag.unique():
        df_ = df.loc[df.Lag==lag]
        x = list(range(len(df_)))
        ax1.plot(x, df_.Bias, label=str(lag))
        ax2.plot(x, df_.RMSE, label=str(lag))
    ax1.set_title('Bias')
    ax2.set_title('RMSE')
    ax1.legend()
    print(f'Saving {figname}')
    fig.savefig(figname)
    plt.close()

def run():
    df = comp_all_errors()
    outfile = 'out/persistence.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print(f'Saving {outfile}')
    df.set_index('Date').to_csv(outfile)
    make_plots(df, 'out/persistence.png')

if __name__ == '__main__':
    run()
