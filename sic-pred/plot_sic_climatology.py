#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from netCDF4 import Dataset

from sic_pred_base import SicPredClimatology

_LATEST_YEAR = 2019
_START = dt.datetime(_LATEST_YEAR,1,1)
_END = dt.datetime(_LATEST_YEAR,1,31)
_NUM_YEARS = 3
_FIG_MASK = f'out/climatology_{_NUM_YEARS}years/sic_%Y%m%d.png'

def plot(sic, dto):
    figname = dto.strftime(_FIG_MASK)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_title(dto.strftime(
        f'{_NUM_YEARS}-year-climatology: %Y-%m-%d'))
    im = ax.imshow(sic, clim=[0,1])
    fig.colorbar(im)
    print(f"Saving {figname}")
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    fig.savefig(figname)
    plt.close()

def run():
    df_all = dict()
    fc = SicPredClimatology(_LATEST_YEAR, _NUM_YEARS)
    dto = _START
    while dto <= _END:
        sic = fc.forecast(dto)
        plot(sic, dto)
        dto += dt.timedelta(1)

if __name__ == '__main__':
    run()
