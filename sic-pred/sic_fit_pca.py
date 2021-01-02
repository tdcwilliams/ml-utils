#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
import pickle

from sic_pred_base import SicPreproc, SicPCA

_START = dt.datetime(2018,1,1)
_END = dt.datetime(2020,12,31)
_OUTFILE1 = 'out/pca/samples.npz'
_OUTFILE2 = 'out/pca/sic_pca.pkl'
#_REF_LAG = 1
_REF_LAG = None

def save_samples():
    days = 1 + (_END - _START).days
    fc = SicPreproc(ref_lag=_REF_LAG)
    datetimes = [_START + dt.timedelta(i) for i in range(days)]
    samples = []
    for dto in datetimes:
        print(f'Getting sample for {dto}')
        samples += [fc.get_sample(dto).flatten()]
    print(f'Saving {_OUTFILE1}')
    np.savez(_OUTFILE1, samples=np.array(samples),
            datetimes=np.array(datetimes), allow_pickle=True)

def load_samples():
    f = np.load(_OUTFILE1, allow_pickle=True)
    return f['samples'], list(f['datetimes'])

def run():
    if not os.path.exists(_OUTFILE1):
        os.makedirs(os.path.dirname(_OUTFILE1), exist_ok=True)
        save_samples()
    sic_pca = SicPCA.init_from_samples(*load_samples(), ref_lag=_REF_LAG)
    print(f'Saving {_OUTFILE2}')
    pickle.dump(sic_pca, open(_OUTFILE2, 'wb'))

if __name__ == '__main__':
    run()
