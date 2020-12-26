#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from netCDF4 import Dataset

from sic_pred_base import SicPredClimatology

_OSISAF_DIR = os.path.join(
        os.getenv('CLUSTER_ROOT_DIR'), 'data/OSISAF_ice_conc/polstere')
_NAME_MASK = os.path.join(_OSISAF_DIR,
        '%Y_nh_polstere/ice_conc_nh_polstere-100_multi_%Y%m%d1200.nc')
_LATEST_YEAR = 2019
_START = dt.datetime(_LATEST_YEAR,1,1)
_END = dt.datetime(_LATEST_YEAR,12,31)
_NUM_YEARS = 4
_FIG_MASK = f'out/climatology_{_NUM_YEARS}years/sic_%Y%m%d.png'

def plot(sic, dto):
    figname = dto.strftime(_FIG_MASK)
    fig = plt.figure((10,10))
    ax = fig.add_subplot(111)
    im = ax.imshow(sic, clim=[0,1], cmap='ice')
    fig.colorbar(im)
    print(f"Saving {figname}")
    fig.savefig(figname)
    plt.close()

def run():
    df_all = dict()
    fc = SicPredClimatology(_LATEST_YEAR, _NUM_YEARS)
    dto = _START
    while dto <= _END:
        dto += dt.timedelta(1)
        sic = fc.forecast(dto)
        plot(sic, dto)

if __name__ == '__main__':
    run()
