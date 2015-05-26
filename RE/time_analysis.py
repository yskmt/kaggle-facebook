"""
time_analysis.py
Facebook Recruiting IV: Human or Robot?
author: Yusuke Sakamoto

Analyze the time-dependent behaviors of the bidders.


"""
from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##########################################################################
# Gather the bid interval counts.
# Use the multiple of the minimum interval (==52631578.95)
##########################################################################

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
# Gather the auction count for the same time-frame
##########################################################################

def gather_sametime_bids_counts(info, bids_by_bidders):
    """Gather the number of bids that occurred at the same time by the
    same bidder for the :
    1. same auction
    2. different auction

    """

    num_bidders = len(info)
    num_bids_sametime = np.zeros((num_bidders, 2), dtype=int)
    nb = 0
    for bidder in info.index:
        if nb % 100 == 0:
            print '%d/%d' % (nb, num_bidders)

        bids = bids_by_bidders[bids_by_bidders['bidder_id'] == bidder]
        bids_time = bids['time']
        bids_time_diff = np.diff(bids_time)

        zero_int = np.where(bids_time_diff == 0)[0]

        bids_sametime_sameauc = 0
        bids_sametime_diffauc = 0
        for i in range(len(zero_int)):
            # print bids.iloc[zero_int[i]]
            # print bids.iloc[zero_int[i]+1]

            if bids.iloc[zero_int[i]]['auction'] == bids.iloc[zero_int[i] + 1]['auction']:
                bids_sametime_sameauc += 1
            else:
                bids_sametime_diffauc += 1

        num_bids_sametime[nb, 0] = bids_sametime_sameauc
        num_bids_sametime[nb, 1] = bids_sametime_diffauc

        nb += 1

    num_bids_sametime \
        = pd.DataFrame(num_bids_sametime, index=info.index)
    num_bids_sametime.columns \
        = ['num_bids_sametime_sameauc', 'num_bids_sametime_diffauc']

    return num_bids_sametime


##########################################################################
# Gather the bid streaks
##########################################################################


def gather_bid_streaks(info, bids, bid_timeframe, max_count=20):
    """
    Gather the bid interval counts.
    Use the multiple of the minimum interval (==52631578.95)
    """

    bid_streaks = []
    for i in range(len(info)):
        if i % 100 == 0:
            print '%d/%d' % (i, len(info))

        bi = plot_intervals_hist(bids, info, info.index[i])

        # normalize by min_interval
        bi = (np.array(bi) + eps) / min_interval

        st = 0
        streaks = []
        for j in range(len(bi)):
            if bi[j] < bid_timeframe:
                st += 1
            elif st > 0:
                streaks.append(st)
                st = 0
            else:
                st = 0

        streaks = pd.DataFrame(
            np.array(sorted(streaks, reverse=True)[:max_count], dtype=int)\
            .reshape(1, min(len(streaks), max_count)),
            index=[info.index[i]], dtype=int)

        bid_streaks.append(streaks)

    bid_streaks = pd.concat(bid_streaks, axis=0)
    bid_streaks.columns = map(lambda x: 'streak_' + str(x), range(max_count))
    bid_streaks.fillna(0, inplace=True)
    
    return bid_streaks


#############################################################################
if __name__ == "__main__":
    bids_bots = pd.read_csv('data/bids_bots.csv')
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)

    bids_humans = pd.read_csv('data/bids_humans.csv')
    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)

    bids_test = pd.read_csv('data/bids_test.csv')
    info_test = pd.read_csv('data/info_test.csv', index_col=0)

    min_interval = 52631578.95
    max_interval = 0.03e12
    eps = 100  # some space for int division to work

    if 'interval-counts' in argv[1]:
        # bots
        bids_intervals_bots = gather_bid_interval_counts(info_bots, bids_bots)
        bids_intervals_bots.to_csv(
            'data/bids_intervals_bots_info.csv', index_label='bidder_id')
        # humans
        bids_intervals_humans = gather_bid_interval_counts(
            info_humans, bids_humans)
        bids_intervals_humans.to_csv(
            'data/bids_intervals_humans_info.csv', index_label='bidder_id')
        # test
        bids_intervals_test = gather_bid_interval_counts(info_test, bids_test)
        bids_intervals_test.to_csv(
            'data/bids_intervals_test_info.csv', index_label='bidder_id')

    elif 'same-time-bids' in argv[1]:
        # humans
        num_bids_sametime_humans = gather_sametime_bids_counts(
            info_humans, bids_humans)
        num_bids_sametime_humans.to_csv(
            'data/num_bids_sametime_info_humans.csv', index_label='bidder_id')
        # bots
        num_bids_sametime_bots = gather_sametime_bids_counts(
            info_bots, bids_bots)
        num_bids_sametime_bots.to_csv(
            'data/num_bids_sametime_info_bots.csv', index_label='bidder_id')
        # test
        num_bids_sametime_test = gather_sametime_bids_counts(
            info_test, bids_test)
        num_bids_sametime_test.to_csv(
            'data/num_bids_sametime_info_test.csv', index_label='bidder_id')

    elif 'bid-streaks' in argv[1]:

        for tf in [40, 80]:
            btf = tf+0.1
            # bots
            bid_streaks_bots = gather_bid_streaks(
                info_bots, bids_bots, bid_timeframe=btf)
            bid_streaks_bots.to_csv(
                'data/bid_streaks_info_bots_%d.csv' %tf, index_label='bidder_id')
            # humans
            bid_streaks_humans = gather_bid_streaks(
                info_humans, bids_humans, bid_timeframe=btf)
            bid_streaks_humans.to_csv(
                'data/bid_streaks_info_humans_%d.csv' %tf, index_label='bidder_id')
            # test
            bid_streaks_test = gather_bid_streaks(
                info_test, bids_test, bid_timeframe=btf)
            bid_streaks_test.to_csv(
                'data/bid_streaks_info_test_%d.csv' %tf, index_label='bidder_id')
        
    elif 'combine_nbs' in argv[1]:

        # combine two columns of the sametime info
        nbsinfo_humans = pd.read_csv('data/num_bids_sametime_info_humans.csv',
                                     index_col=0)
        nbsinfo_bots = pd.read_csv('data/num_bids_sametime_info_bots.csv',
                                   index_col=0)
        nbsinfo_test = pd.read_csv('data/num_bids_sametime_info_test.csv',
                                   index_col=0)

        nbsinfo_humans = pd.concat([nbsinfo_humans, nbsinfo_humans['num_bids_sametime_sameauc']+nbsinfo_humans['num_bids_sametime_diffauc']], axis=1)
        nbsinfo_humans.columns = [u'num_bids_sametime_sameauc',
                                  u'num_bids_sametime_diffauc', 'num_bids_sametime']

        nbsinfo_bots = pd.concat([nbsinfo_bots, nbsinfo_bots['num_bids_sametime_sameauc']+nbsinfo_bots['num_bids_sametime_diffauc']], axis=1)
        nbsinfo_bots.columns = [u'num_bids_sametime_sameauc',
                                  u'num_bids_sametime_diffauc', 'num_bids_sametime']

        nbsinfo_test = pd.concat([nbsinfo_test, nbsinfo_test['num_bids_sametime_sameauc']+nbsinfo_test['num_bids_sametime_diffauc']], axis=1)
        nbsinfo_test.columns = [u'num_bids_sametime_sameauc',
                                  u'num_bids_sametime_diffauc', 'num_bids_sametime']

        nbsinfo_humans.to_csv(
            'data/num_bids_sametime_info_humans.csv', index_label='bidder_id')
        nbsinfo_bots.to_csv(
            'data/num_bids_sametime_info_bots.csv', index_label='bidder_id')
        nbsinfo_test.to_csv(
            'data/num_bids_sametime_info_test.csv', index_label='bidder_id')

    elif 'combine_bstr' in argv[1]:
        bstrcomb_humans = []
        bstrcomb_bots = []
        bstrcomb_test = []
        
        # combine bid streak data: concat only the first column (longest streak)
        tfs = [1, 5, 10, 15, 20, 40, 80]
        for bs in tfs:
            print bs
            bstrinfo_humans = pd.read_csv('data/bid_streaks_info_humans_%d.csv' %bs,
                                         index_col=0)
            bstrinfo_bots = pd.read_csv('data/bid_streaks_info_bots_%d.csv' %bs,
                                       index_col=0)
            bstrinfo_test = pd.read_csv('data/bid_streaks_info_test_%d.csv' %bs,
                                       index_col=0)

            bstrcomb_humans.append(bstrinfo_humans.iloc[:,0])
            bstrcomb_bots.append(bstrinfo_bots.iloc[:,0])
            bstrcomb_test.append(bstrinfo_test.iloc[:,0])

        bstrcomb_humans = pd.concat(bstrcomb_humans, axis=1)
        bstrcomb_bots = pd.concat(bstrcomb_bots, axis=1)
        bstrcomb_test = pd.concat(bstrcomb_test, axis=1)

        bstrcomb_humans.columns = map(lambda x: 'streak_'+ str(x), tfs)
        bstrcomb_bots.columns = map(lambda x: 'streak_'+ str(x), tfs)
        bstrcomb_test.columns = map(lambda x: 'streak_'+ str(x), tfs)

        bstrcomb_humans.to_csv('data/max_streak_info_humans.csv',
                               index_label='bidder_id')
        bstrcomb_bots.to_csv('data/max_streak_info_bots.csv',
                               index_label='bidder_id')
        bstrcomb_test.to_csv('data/max_streak_info_test.csv',
                               index_label='bidder_id')
