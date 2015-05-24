"""
analysis_funcs.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd

def gather_info(bids_data):
    """
    Gather the useful infromation from bids data.
    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    # tmp = np.zeros((num_bidders, 7), dtype=int)
    # tmp_mch = np.zeros(num_bidders, dtype=object)

    bidders_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # number of bids by this bidder
        num_bids = len(bids)
        # number of auction by this bidder
        num_aucs = len(bids['auction'].unique())
        # number of merchandises by this bidder
        num_merchs = len(bids['merchandise'].unique())
        # number of devices used by this bidder
        num_devices = len(bids['device'].unique())
        # number of countries by this bidder
        num_countries = len(bids['country'].unique())
        # number of ips by this bidder
        num_ips = len(bids['ip'].unique())
        # number of urls by this bidder
        num_urls = len(bids['url'].unique())

        # primary merchandise. NOTE: every bidder only has one merchandise
        merch = bids['merchandise'].unique()[0]

        tmp_info = np.array([num_bids, num_aucs, num_merchs,
                             num_devices, num_countries, num_ips, num_urls,
                             merch])

        tmp_info = tmp_info.reshape(1, len(tmp_info))

        bidders_info.append(pd.DataFrame(
            tmp_info,  index=[bidder_ids[i]],
            columns=['num_bids', 'num_aucs', 'num_merchs', 'num_devices',
                     'num_countries', 'num_ips', 'num_urls', 'merchandise']
        ))

    bidders_info = pd.concat(bidders_info, axis=0)
    bidders_info.index.name = 'bidder_id'

    return bidders_info


def gather_auc_bids_info(bids_data):
    """
    Gather the number of bids for each auction info.
    The number of bids are sorted in descending order.
    """

    # ANALYSIS
    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    # for each bidder
    bidders_aucbids_info = []
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # count number of bids for each auction by this bidder
        num_bids_auction = []
        for auction in bids['auction'].unique():
            num_bids_auction.append(len(bids[bids['auction'] == auction]))

        bidders_aucbids_info.append(sorted(num_bids_auction, reverse=True))

    bbainfo_bots = pd.DataFrame(bidders_aucbids_info, index=bidder_ids)
    bbainfo_bots.fillna(0, inplace=True)

    # change column label to reflect the number of bids and add prefix
    bbainfo_bots.columns = map(lambda x: 'bba_' + str(x),
                               range(1, bbainfo_bots.shape[1] + 1))

    return bbainfo_bots


def gather_country_info(bids_data):
    """
    Gather the country infromation from bids data.
    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    bidders_country_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # number of occurences of each country by this bidder
        bidders_country_info.append(bids['country'].value_counts())

        pd.concat([pd.DataFrame(bidders_country_info[0]).transpose(),
                   pd.DataFrame([bidder_ids[0]])], axis=1)

    bc_info = pd.concat(bidders_country_info, axis=1).transpose()
    bidders_country_info \
        = pd.concat([bc_info,
                     pd.DataFrame(bidder_ids, columns=['bidder_id'])], axis=1)\
        .set_index('bidder_id')

    bidders_country_info.fillna(0)

    return bidders_country_info


def gather_device_info(bids_data):
    """
    Gather the device infromation from bids data.
    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    bidders_device_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # number of occurences of each device by this bidder
        bidders_device_info.append(bids['device'].value_counts())

        pd.concat([pd.DataFrame(bidders_device_info[0]).transpose(),
                   pd.DataFrame([bidder_ids[0]])], axis=1)

    bd_info = pd.concat(bidders_device_info, axis=1).transpose()
    bidders_device_info \
        = pd.concat([bd_info,
                     pd.DataFrame(bidder_ids, columns=['bidder_id'])], axis=1)\
        .set_index('bidder_id')

    bidders_device_info.fillna(0)

    return bidders_device_info



def gather_count_info(bids_data, item):
    """
    Gather the number of count that item occurs for the same bidder.

    bids_data: bids data by humans, bots, or test
    item: column label (device, country, etc)

    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    bidders_count_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # number of occurences of each unique item by this bidder
        bidders_count_info.append(bids[item].value_counts())

        pd.concat([pd.DataFrame(bidders_count_info[0]).transpose(),
                   pd.DataFrame([bidder_ids[0]])], axis=1)

    bd_info = pd.concat(bidders_count_info, axis=1).transpose()
    bidders_count_info \
        = pd.concat([bd_info,
                     pd.DataFrame(bidder_ids, columns=['bidder_id'])], axis=1)\
        .set_index('bidder_id')

    bidders_count_info.fillna(0)

    return bidders_count_info
