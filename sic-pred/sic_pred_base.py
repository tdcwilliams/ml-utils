#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pandas as pd
from collections import defaultdict
from netCDF4 import Dataset
from matplotlib import pyplot as plt, dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_OSISAF_DIR = os.path.join(
        os.getenv('CLUSTER_ROOT_DIR'), 'data/OSISAF_ice_conc/polstere')
_NAME_MASK = os.path.join(_OSISAF_DIR,
        '%Y_nh_polstere/ice_conc_nh_polstere-100_multi_%Y%m%d1200.nc')
_MASK_VARS = dict(np.load('out/OSISAF_medium_arctic_mask.npz'))
_AREA_FACTOR = 10e3*10e3/(1e6*1e6) #grid cell area/10^6km^2

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
    def get_ice_mask(sic):
        ice = np.zeros_like(sic)
        ice[sic>.15] = 1
        ice[np.isnan(sic)] = np.nan
        return ice

    def comp_errors(self, sic, sic_hat):
        errors = dict()
        errors['Bias'] = np.nanmean(sic_hat-sic)
        errors['RMSE'] = np.sqrt(np.nanmean((sic_hat-sic)**2))
        diff = self.get_ice_mask(sic_hat) - self.get_ice_mask(sic)
        errors['Extent_Bias'] = _AREA_FACTOR*np.nansum(diff)
        errors['IIEE'] = _AREA_FACTOR*np.nansum(np.abs(diff))
        return errors

    @staticmethod
    def map_errors(sic, sic_hat, dto, figname):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,10))
        for ax, arr, clim, cmap, ttl in zip(axes,
                [sic, sic_hat, sic_hat-sic],
                [(0,1),(0,1),(-.5,.5)],
                ['viridis', 'viridis', 'bwr'],
                ['SIC Obs.', 'SIC Model', 'SIC Bias'],
                ):
            im = ax.imshow(arr, clim=clim, cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=.4)
            ax.set_title(ttl + dto.strftime(' %Y-%m-%d'))
        if os.path.sep in figname:
            os.makedirs(os.path.dirname(figname), exist_ok=True)
        print(f'Saving {figname}')
        fig.savefig(figname)
        plt.close()

    def comp_all_errors(self, start, end, figmask=None):
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
            if figmask is not None:
                self.map_errors(sic, sic_hat, dto, dto.strftime(figmask))
        return pd.DataFrame(errors)

    @staticmethod
    def plot_errors(df_all, figname):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
        for lbl, df in df_all.items():
            for ax, stat, units in zip(axes.flatten(),
                    ['Bias', 'RMSE', 'Extent_Bias', 'IIEE'],
                    ['', '', r', $10^6$km$^2$', r', $10^6$km$^2$'],
                    ):
                ax.plot(df.Date, df[stat], label=lbl)
                ax.set_ylabel(stat.replace('_', ' ') + units)
                ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[0,0].legend()
        fig.autofmt_xdate()
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        print(f'Saving {figname}')
        fig.savefig(figname)
        plt.close()

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

    def __init__(self, ref_lag=None):
        self.ref_lag = ref_lag

    def get_sample(self, dto):
        arr = self.get_conc(dto)
        if self.ref_lag is not None:
            arr -= self.get_conc(dto - dt.timedelta(self.ref_lag))
        return np.array([arr[np.isfinite(arr)]])

    def map_to_grid(self, sample):
        mask = self.mask
        v = np.zeros(mask.shape)
        v[~mask] = sample.flatten()
        v[mask] = np.nan
        return v

    def convert_sample(self, dto, sample):
        out = self.map_to_grid(sample)
        if self.ref_lag is not None:
            out += self.get_conc(dto - dt.timedelta(self.ref_lag))
        return out

    @staticmethod
    def get_scaler(samples):
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        return scaler, samples

class SicPCA(SicPreproc):
    def __init__(self, pca, scaler, datetimes, ref_lag=None):
        super().__init__(ref_lag=ref_lag)
        self.pca = pca
        self.scaler = scaler
        self.datetimes = datetimes
        self.transformers = [self.scaler, self.pca]

    @classmethod
    def init_from_samples(cls, samples, datetimes, ref_lag=None, **kwargs):
        scaler, scaled_samples = cls.get_scaler(samples)
        pca = PCA(**kwargs)
        pca.fit(scaled_samples)
        return SicPCA(pca, scaler, datetimes, ref_lag=ref_lag)

    def get_component(self, i):
        return self.pca.components_[i]

    def transform(self, sample):
        output = np.copy(sample)
        for obj in self.transformers:
            output = obj.transform(output)
        return output

    def inverse_transform(self, transform):
        output = np.copy(transform)
        for obj in self.transformers[::-1]:
            output = obj.inverse_transform(output)
        return output

    def project(self, dto, n_components=None):
        sample = self.get_sample(dto)
        transform = self.transform(sample)
        if n_components is not None:
            transform[0,n_components:] = 0.
        return self.convert_sample(dto, self.inverse_transform(transform))

    def comp_all_errors(self, start, end, n_components=None, figmask=None):
        days = 1 + (end - start).days
        errors = defaultdict(list)
        for i in range(days):
            dto = start + dt.timedelta(i)
            print(dto)
            sic = self.get_conc(dto)
            sic_hat = self.project(dto, n_components=n_components)
            errors['Date'] += [dto]
            for k, v in self.comp_errors(sic, sic_hat).items():
                errors[k] += [v]
            if figmask is not None:
                self.map_errors(sic, sic_hat, dto, dto.strftime(figmask))
        return pd.DataFrame(errors)
