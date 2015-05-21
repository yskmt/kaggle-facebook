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


##########################################################################
# Gather the bid interval counts.
# Use the multiple of the minimum interval (==52631578.95)
##########################################################################
# bots
bids_intervals_bots = []
for i in range(len(info_bots)):
    if i % 100 == 0:
        print '%d/%d' % (i, len(info_bots))

    bi = plot_intervals_hist(bids_bots, info_bots, info_bots.index[i])

    # normalize by min_interval
    bi = (np.array(bi) + eps) / min_interval
    # use the first 100 interval unit
    bi_df = pd.DataFrame(
        np.histogram(bi[bi < 101], bins=range(0, 101))[0].reshape(1, 100),
        index=[info_bots.index[i]])

    bids_intervals_bots.append(bi_df)

bids_intervals_bots = pd.concat(bids_intervals_bots, axis=0)
bids_intervals_bots.to_csv(
    'data/bids_intervals_bots_info.csv', index_label='bidder_id')

##########################################################################
# humans
bids_intervals_humans = []
for i in range(len(info_humans)):
    if i % 100 == 0:
        print '%d/%d' % (i, len(info_humans))

    bi = plot_intervals_hist(bids_humans, info_humans, info_humans.index[i])
    # bids_intervals_humans += list(bi)

    # normalize by min_interval
    bi = (np.array(bi) + eps) / min_interval
    # use the first 100 interval unit
    bi_df = pd.DataFrame(
        np.histogram(bi[bi < 101], bins=range(0, 101))[0].reshape(1, 100),
        index=[info_humans.index[i]])

    bids_intervals_humans.append(bi_df)

bids_intervals_humans = pd.concat(bids_intervals_humans, axis=0)
bids_intervals_humans.to_csv(
    'data/bids_intervals_humans_info.csv', index_label='bidder_id')

##########################################################################
# test
bids_intervals_test = []
for i in range(len(info_test)):
    if i % 100 == 0:
        print '%d/%d' % (i, len(info_test))

    bi = plot_intervals_hist(bids_test, info_test, info_test.index[i])
    # bids_intervals_test += list(bi)

    # normalize by min_interval
    bi = (np.array(bi) + eps) / min_interval
    # use the first 100 interval unit
    bi_df = pd.DataFrame(
        np.histogram(bi[bi < 101], bins=range(0, 101))[0].reshape(1, 100),
        index=[info_test.index[i]])

    bids_intervals_test.append(bi_df)

bids_intervals_test = pd.concat(bids_intervals_test, axis=0)
bids_intervals_test.to_csv(
    'data/bids_intervals_test_info.csv', index_label='bidder_id')


##########################################################################
# sdf

# zero_int_humans = []
# bids_intervals_humans = []
# for i in range(len(info_humans)):
#     bi = plot_intervals_hist(bids_humans, info_humans, info_humans.index[i])
#     bids_intervals_humans += list(bi)
#     zero_int_humans.append(sum(bi < 0.1))


# eps = 100

# bih = (np.array(sorted(bids_intervals_humans)) + eps) / min_interval
# bib = (np.array(sorted(bids_intervals_bots)) + eps) / min_interval
# bih = bih[bih < 100]
# bib = bib[bib < 100]
# plt.hist([bih, bib], bins=100,
#          normed=True, label=['humans', 'bots'])
# plt.legend(loc='best')
# plt.show()


# print plot_intervals_hist(bids_humans, info_humans, info_humans.index[1872])

# sdf
# # bidstime -= bidstime[0]
# # bidstime /= 1e10

# # print bidstime[0:-1]


# bids_humans = pd.read_csv('data/bids_humans.csv')
# info_humans = pd.read_csv('data/info_humans.csv', index_col=0)

# for idx in info_humans.index:
#     print idx
#     bh = bids_humans[bids_humans['bidder_id'] == idx]
#     bh['time'].values
#     bidstime /= 1e10

#     if len(bidstime) > 100:
#         print bidstime

# # extract the "bid frequency" data
# # objective: the interval between one bid to the other and generate a distribution.
# # create a bin and categorize them into ~10 intervals?
# # idea: bots should have more bids with small intervals than humans.
