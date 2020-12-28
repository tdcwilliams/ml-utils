#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sic_pred_base import SicPredBase

_START = dt.datetime(2020,1,1)
_END = dt.datetime(2020,12,21)
_OUTFILES = dict(
        scaler = 'out/pca/scaler.pkl',
        pca = 'out/pca/pca.pkl'
        )

class SicPCA(SicPredBase):
    
    def get_ref_conc(self, dto):
        return self.get_conc(dto - dt.timedelta(1))

    def get_sample(self, dto):
        dsic = self.get_conc(dto) - self.get_ref_conc(dto)
        gpi = np.isfinite(dsic)
        return dsic[gpi]

    def convert_sample(self, dto, sample):
        sic = self.get_ref_conc(dto)
        gpi = np.isfinite(sic)
        sic[gpi] += sample
        return sic
            
def run():
    days = 1 + (_END - _START).days
    fc = SicPCA()
    samples = []
    for i in range(days):
        dto = _START + dt.timedelta(i)
        print(f'Getting sample for {dto}')
        print(fc.get_sample(dto).shape)
        samples += [fc.get_sample(dto)]
    samples = np.array(samples)
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)
    pca = PCA()
    principal_components = pca.fit_transform(samples)

    # save to file
    os.makedirs('out/pca', exist_ok=True)
    for v,k in [(scaler, 'scaler'), (pca, 'pca')]:
        f =  _OUTFILES[k]
        print(f'Saving {f}')
        pickle.dump(v, open(f, 'wb'))

if __name__ == '__main__':
    run()
