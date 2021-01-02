#! /usr/bin/env python3
import os
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pandas as pd

from sic_pred_base import SicPredClimatology

_START = dt.datetime(2020,1,1)
_END = dt.datetime(2020,12,21)
_LATEST_YEAR = 2019
_MAX_YEARS = 3

def run():
    df_all = dict()
    for num_years in range(1, _MAX_YEARS + 1):
        fc = SicPredClimatology(_LATEST_YEAR, num_years)
        outfile = f'out/climatology-years{num_years}.csv'
        if os.path.exists(outfile):
            print(f'Reading {outfile}')
            df = pd.read_csv(outfile)
        else:
            df = fc.comp_all_errors(_START, _END)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            print(f'Saving {outfile}')
            df.set_index('Date').to_csv(outfile)
        df_all[f'Years={num_years}'] = df
    SicPredClimatology.plot_errors(df_all, 'out/climatology.png')

if __name__ == '__main__':
    run()
