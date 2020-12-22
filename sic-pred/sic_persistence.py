#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict

from netCDF4 import Dataset

_OSISAF_DIR = os.path.join(
        os.getenv('CLUSTER_ROOT_DIR'), 'data/OSISAF_ice_conc/polstere')
_NAME_MASK = os.path.join(_OSISAF_DIR,
        '%Y_nh_polstere/ice_conc_nh_polstere-100_multi_%Y%m%d1200.nc')

def get_conc(f):
    #print(f'Reading SIC from {f}')
    with Dataset(f, 'r') as ds:
        return .01*ds.variables['ice_conc'][0].filled(np.nan)

def get_prediction(dto, lag):
    dto_hat = dto - dt.timedelta(lag)
    return get_conc(dto_hat.strftime(_NAME_MASK))

def comp_errors(dto, lag):
    errors = dict()
    sic = get_conc(dto.strftime(_NAME_MASK))
    sic_hat = get_prediction(dto, lag)
    errors = dict()
    errors['Bias'] = np.nanmean(sic_hat-sic)
    errors['RMSE'] = np.sqrt(np.nanmean((sic_hat-sic)**2))
    return errors

def comp_all_errors():
    start = dt.datetime(2018,8,1)
    end = dt.datetime(2018,8,14)
    #end = dt.datetime(2019,7,31)
    days = 1 + (end - start).days
    dates = [start + dt.timedelta(i) for i in range(days)]
    lags = [i+1 for i in range(7)]
    errors = defaultdict(list)
    for dto, lag in itertools.product(dates, lags):
        errors['Date'] += [dto]
        errors['Lag'] += [lag]
        for k, v in comp_errors(dto, lag).items():
            errors[k] += [v]
    return pd.DataFrame(errors)

def run():
    df = comp_all_errors()
    outfile = 'out/persistence.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print(f'Saving {outfile}')
    df.set_index('Date').to_csv(outfile)

if __name__ == '__main__':
    run()
