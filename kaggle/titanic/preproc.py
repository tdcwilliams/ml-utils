#! /usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict

# rules to extract features from raw data
def proc_names(names):
    """ extract title (eg 'Mrs.') from full name """
    d = defaultdict(list)
    for name in names:
        ttl = name.split(', ')[1].removeprefix('the ').split()[0]
        d['Title'] += [ttl]
    return pd.DataFrame(data=d, index=names.index)

def proc_tickets(tickets):
    """
    extract ticket tag from ticket reference
    eg 347082 -> 'None'
    eg STON/O2. 3101282 -> STON/O2.
    """
    d = defaultdict(list)
    for ticket in tickets:
        s = ticket.split()
        if len(s) == 1:
            d['TicketTag'] += ['nan']
        else:
            d['TicketTag'] += s[:1]
    return pd.DataFrame(data=d, index=tickets.index)

def proc_cabins(cabins):
    """ extract cabin tag and number of cabins from cabin """
    d = defaultdict(list)
    for cabin in cabins.values.astype(str):
        if cabin == 'nan':
            d['CabinTag'] += ['nan']
            d['NumCabins'] += [0]
            continue
        s = cabin.split()
        d['NumCabins'] += [len(s)]
        d['CabinTag'] += [s[0][0]]
    return pd.DataFrame(data=d, index=cabins.index)

_MAPPINGS = defaultdict(lambda : lambda x:x, {
    'Name'   : proc_names,
    'Ticket' : proc_tickets,
    'Cabin'  : proc_cabins,
    })

def proc_csv(f):
    """ Process 1 csv file to return a DataFrame with features """
    df = pd.read_csv(f, index_col = 'PassengerId')
    features = pd.DataFrame(index=df.index)
    for col in df.columns:
        mapping = _MAPPINGS[col]
        features = features.join(mapping(df[col]))
    return features

if __name__ == '__main__':
    for f in ['train.csv', 'test.csv']:
        ofil = f'features.{f}'
        print(f'Saving {ofil}')
        features = proc_csv(f)
        features.to_csv(ofil)
