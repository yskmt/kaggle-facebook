"""
time_analysis.py
Facebook Recruiting IV: Human or Robot?
author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bids_bots = pd.read_csv('data/bids_bots.csv')
info_bots = pd.read_csv('data/info_bots.csv', index_col=0)

bids_humans = pd.read_csv('data/bids_humans.csv')
info_humans = pd.read_csv('data/info_humans.csv', index_col=0)

bids_test = pd.read_csv('data/bids_test.csv')
info_test = pd.read_csv('data/info_test.csv', index_col=0)


min_interval = 52631578.95
max_interval = 0.03e12
eps = 100  # some space for int division to work


def plot_intervals_hist(bids, info, bidder_id, plot=False):

    bidstime = bids[bids['bidder_id'] == bidder_id]['time'].values
    bids_intervals = np.diff(bidstime)
    # bids_intervals = np.array(sorted(bids_intervals))
    # bids_intervals = bids_intervals[bids_intervals < max_interval]

    if plot:
        n, bins, patches = plt.hist(bids_intervals,
                                    bins=50,
                                    histtype='stepfilled')
        plt.show()

    return bids_intervals


def gather_bid_interval_counts(info, bids):
    """
    Gather the bid interval counts.
    Use the multiple of the minimum interval (==52631578.95)
    """

    bids_intervals = []
    for i in range(len(info)):
        if i % 100 == 0:
            print '%d/%d' % (i, len(info))

        bi = plot_intervals_hist(bids, info, info.index[i])

        # normalize by min_interval
        bi = (np.array(bi) + eps) / min_interval
        # use the first 100 interval unit
        bi_df = pd.DataFrame(
            np.histogram(bi[bi < 101], bins=range(0, 101))[0].reshape(1, 100),
            index=[info.index[i]])

        bids_intervals.append(bi_df)

    bids_intervals = pd.concat(bids_intervals, axis=0)
    bids_intervals.columns = map(lambda x: 'int_' + str(x), range(100))

    return bids_intervals


##########################################################################
# Gather the bid interval counts.
# Use the multiple of the minimum interval (==52631578.95)
##########################################################################
# bots
bids_intervals_bots = gather_bid_interval_counts(info_bots, bids_bots)
bids_intervals_bots.to_csv(
    'data/bids_intervals_bots_info.csv', index_label='bidder_id')
# humans
bids_intervals_humans = gather_bid_interval_counts(info_humans, bids_humans)
bids_intervals_humans.to_csv(
    'data/bids_intervals_humans_info.csv', index_label='bidder_id')
# test
bids_intervals_test = gather_bid_interval_counts(info_test, bids_test)
bids_intervals_test.to_csv(
    'data/bids_intervals_test_info.csv', index_label='bidder_id')
