#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pandas as pd
from collections import defaultdict
from netCDF4 import Dataset

_OSISAF_DIR = os.path.join(
        os.getenv('CLUSTER_ROOT_DIR'), 'data/OSISAF_ice_conc/polstere')
_NAME_MASK = os.path.join(_OSISAF_DIR,
        '%Y_nh_polstere/ice_conc_nh_polstere-100_multi_%Y%m%d1200.nc')

class SicPredBase:

    @staticmethod
    def get_conc(dto):
        #print(f'Reading SIC from {f}')
        f = dto.strftime(_NAME_MASK)
        with Dataset(f, 'r') as ds:
            return .01*ds.variables['ice_conc'][0].filled(np.nan)

    @staticmethod
    def comp_errors(sic, sic_hat):
        errors = dict()
        errors['Bias'] = np.nanmean(sic_hat-sic)
        errors['RMSE'] = np.sqrt(np.nanmean((sic_hat-sic)**2))
        return errors

    def comp_all_errors(self, start, end):
        days = 1 + (end - start).days
        errors = defaultdict(list)
        for i in range(days):
            dto = start + dt.timedelta(i)
            print(dto)
            sic = self.get_conc(dto)
            sic_hat = self.forecast(dto)
            errors['Date'] += [dto]
            for k, v in self.comp_errors(sic, sic_hat).items():
                errors[k] += [v]
        return pd.DataFrame(errors)

class SicPredPersistence(SicPredBase):
    def __init__(self, lag):
        self.lag = lag

    def forecast(self, dto):
        return self.get_conc(dto - dt.timedelta(self.lag))

class SicPredClimatology(SicPredBase):
    def __init__(self, latest_year, num_years):
        self.latest_year = latest_year
        self.num_years = num_years

    def forecast(self, dto):
        wt = 1/self.num_years
        sic_hat = 0
        for i in range(self.num_years):
            dto_i = dt.datetime(self.latest_year - i, dto.month, dto.day)
            sic_hat += wt*self.get_conc(dto_i)
        return self.get_conc(dto - dt.timedelta(self.lag))

