#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pandas as pd

from sic_pred_base import SicPredPersistence

_START = dt.datetime(2018,1,20)
_END = dt.datetime(2020,12,31)
_MAX_LAG = 7

def run():
    df_all = dict()
    for lag in range(1, _MAX_LAG + 1):
        outfile = f'out/persistence-lag{lag}.csv'
        if not os.path.exists(outfile):
            fc = SicPredPersistence(lag)
            df = fc.comp_all_errors(_START, _END)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            print(f'Saving {outfile}')
            df.set_index('Date').to_csv(outfile)
        print(f'Reading {outfile}')
        df = pd.read_csv(outfile)
        df_all[f'Lag={lag}'] = df
    SicPredPersistence.plot_errors(df_all, 'out/persistence.png')

if __name__ == '__main__':
    run()
