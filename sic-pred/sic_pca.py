#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pickle

from sic_pred_base import SicPreproc, SicPCA

_START = dt.datetime(2020,1,1)
_END = dt.datetime(2020,12,21)
_OUTFILE = 'out/pca/pca.pkl'

def load_samples():
    days = 1 + (_END - _START).days
    fc = SicPreproc()
    samples = []
    for i in range(days):
        dto = _START + dt.timedelta(i)
        print(f'Getting sample for {dto}')
        print(fc.get_sample(dto).shape)
        samples += [fc.get_sample(dto)]
    return np.array(samples)

def run():
    samples = load_samples()
    pca = SicPCA.init_from_samples(samples)
    print(f'Saving {_OUTFILE}')
    os.makedirs(os.path.dirname(_OUTFILE), exist_ok=True)
    pickle.dump(pca, open(_OUTFILE, 'wb'))

if __name__ == '__main__':
    run()
