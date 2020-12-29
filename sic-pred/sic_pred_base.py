#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pandas as pd
from collections import defaultdict
from netCDF4 import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_OSISAF_DIR = os.path.join(
        os.getenv('CLUSTER_ROOT_DIR'), 'data/OSISAF_ice_conc/polstere')
_NAME_MASK = os.path.join(_OSISAF_DIR,
        '%Y_nh_polstere/ice_conc_nh_polstere-100_multi_%Y%m%d1200.nc')
_MASK_VARS = dict(np.load('out/OSISAF_medium_arctic_mask.npz'))

class SicPredBase:

    @staticmethod
    def get_conc(dto):
        if dto < dt.datetime(2016,6,1):
            raise ValueError('Ignoring data pre-2016-06-01 due to missing data mask')
        #print(f'Reading SIC from {f}')
        f = dto.strftime(_NAME_MASK)
        i0, i1, j0, j1 = _MASK_VARS['bbox']
        with Dataset(f, 'r') as ds:
            sic = .01*ds.variables['ice_conc'][0][i0:i1, j0:j1].filled(np.nan)
        sic[_MASK_VARS['mask']] = np.nan
        return sic

    @property
    def mask(self):
        return np.isnan(self.get_conc(dt.datetime(2018,1,1)))

    @staticmethod
    def comp_errors(sic, sic_hat):
        errors = dict()
        errors['Bias'] = np.nanmean(sic_hat-sic)
        errors['RMSE'] = np.sqrt(np.nanmean((sic_hat-sic)**2))
        return errors

    def comp_all_errors(self, start, end, **kwargs):
        days = 1 + (end - start).days
        errors = defaultdict(list)
        for i in range(days):
            dto = start + dt.timedelta(i)
            print(dto)
            sic = self.get_conc(dto)
            sic_hat = self.forecast(dto, **kwargs)
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
        i0 = dto.year - self.latest_year
        assert(i0 >= 0)
        for i in range(i0, i0 + self.num_years):
            dto_i = dto - dt.timedelta(int(i*365.25))
            sic_hat += wt*self.get_conc(dto_i)
        return sic_hat

class SicPreproc(SicPredBase):

    def get_ref_conc(self, dto):
        return self.get_conc(dto - dt.timedelta(1))

    def get_sample(self, dto):
        dsic = self.get_conc(dto) - self.get_ref_conc(dto)
        gpi = np.isfinite(dsic)
        return dsic[gpi]

    def map_to_grid(self, sample):
        mask = self.mask
        v = np.zeros(mask.shape)
        v[~mask] = sample
        v[mask] = np.nan
        return v

    def convert_sample(self, dto, sample):
        return self.get_ref_conc(dto) + self.map_to_grid(sample)

    @staticmethod
    def get_scaler(samples):
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        return scaler, samples

class SicPCA(SicPreproc):
    def __init__(self, pca, scaler, datetimes):
        self.pca = pca
        self.scaler = scaler
        self.datetimes = datetimes

    @classmethod
    def init_from_samples(cls, samples, datetimes, **kwargs):
        scaler, scaled_samples = cls.get_scaler(samples)
        pca = PCA(**kwargs)
        pca.fit(scaled_samples)
        return SicPCA(pca, scaler, datetimes)

    def inverse_transforms(transform):
        output = np.copy(transform)
        for obj in [self.pca, self.scaler]:
            output = obj.inverse_transform(output)
        return output

    def get_component(self, i):
        print(self.pca.components_[i,:10])
        return self.scaler.inverse_transform(self.pca.components_[i])

    def forecast(self, dto, index_components=None):
        i =  self.datetimes.index(dto)
        sample = np.zeros((1, self.pca.n_features_))
        if index_components is not None:
            index_components = slice(None)
        sample[index_components] = self.pca.principal_components_[i, index_components]
        return self.convert_sample(dto, self.inverse_transforms(sample))
