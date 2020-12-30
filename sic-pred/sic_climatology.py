#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

from sic_pred_base import SicPredClimatology

_START = dt.datetime(2020,1,1)
_END = dt.datetime(2020,12,21)
_LATEST_YEAR = 2019
_MAX_YEARS = 3

def make_plots(df_all, figname):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
    for num_years, df in df_all.items():
        for ax, stat, units in zip(axes.flatten(),
                ['Bias', 'RMSE', 'Extent_Bias', 'IIEE'],
                ['', '', r', $10^6$km$^2$', r', $10^6$km$^2$'],
                ):
            ax.plot(df.Date, df[stat], label='Years=%i' %num_years)
            ax.set_ylabel(stat.replace('_', ' ') + units)
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0,0].legend()
    fig.autofmt_xdate()
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print(f'Saving {figname}')
    fig.savefig(figname)
    plt.close()

def run():
    df_all = dict()
    for num_years in range(1, _MAX_YEARS + 1):
        fc = SicPredClimatology(_LATEST_YEAR, num_years)
        df = fc.comp_all_errors(_START, _END)
        outfile = f'out/climatology-years{num_years}.csv'
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        print(f'Saving {outfile}')
        df.set_index('Date').to_csv(outfile)
        df_all[num_years] = df
    make_plots(df_all, 'out/climatology.png')

if __name__ == '__main__':
    run()
