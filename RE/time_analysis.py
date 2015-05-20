"""
time_analysis.py
Facebook Recruiting IV: Human or Robot?
author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd
import pylab as P

bids_bots = pd.read_csv('data/bids_bots.csv')
info_bots = pd.read_csv('data/info_bots.csv', index_col=0)

bids_humans = pd.read_csv('data/bids_humans.csv')
info_humans = pd.read_csv('data/info_humans.csv', index_col=0)

min_interval = 52631578

def plot_intervals_hist(bids, info, bidder_id):

    bidstime = bids[bids['bidder_id']==bidder_id]['time'].values
    bids_intervals = np.diff(bidstime)
    bids_intervals = np.array(sorted(bids_intervals))/min_interval

    bi_max = np.max(bids_intervals)

    max_idx = sum(np.array(bids_intervals)<100)

    # max_idx = 200
    n, bins, patches = P.hist(bids_intervals[:max_idx], 50,
                              histtype='stepfilled')
    # P.show()

    return bids_intervals


for i in range(len(info_bots)):
    try:
        bi = plot_intervals_hist(bids_bots, info_bots, info_bots.index[i])
        print bi[:100]
    except:
        print "error", i
        pass
    
for i in range(len(info_humans)):
    if (info_humans.iloc[i].num_bids)>200:
        bi = plot_intervals_hist(bids_humans, info_humans, info_humans.index[i])
        print bi[:100]


print plot_intervals_hist(bids_humans, info_humans, info_humans.index[1872])

sdf
# bidstime -= bidstime[0]
# bidstime /= 1e10

# print bidstime[0:-1]


bids_humans = pd.read_csv('data/bids_humans.csv')
info_humans = pd.read_csv('data/info_humans.csv', index_col=0)

for idx in info_humans.index:
    print idx
    bh = bids_humans[bids_humans['bidder_id']==idx]
    bh['time'].values
    bidstime /= 1e10

    if len(bidstime)>100:
        print bidstime

# extract the "bid frequency" data
# objective: the interval between one bid to the other and generate a distribution.
# create a bin and categorize them into ~10 intervals?
# idea: bots should have more bids with small intervals than humans.


