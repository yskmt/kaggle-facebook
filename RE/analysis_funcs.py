"""
analysis_funcs.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd
from pdb import set_trace


auction_periods = np.array([[9.62, 9.66], [9.68, 9.72], [9.74, 9.78]]) * 1e15


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


def gather_bayes_counts(bids_data):
    """Gather the useful infromation: bots and human counts for each
    unique merchandise, country, device, url, ip labels

    """

    merchandises = bids_data['merchandise'].unique()
    merchandise_counts = []
    for merchandise in merchandises:
        merchandise_counts.append(
            len(bids_data[bids_data['merchandise'] == merchandise]['bidder_id'].unique()))
    df_merchandise = pd.DataFrame(
        np.array(merchandise_counts).reshape(1, len(merchandise_counts)), columns=merchandises)

    countries = bids_data['country'].unique()
    country_counts = []
    for country in countries:
        country_counts.append(
            len(bids_data[bids_data['country'] == country]['bidder_id'].unique()))
    df_country = pd.DataFrame(
        np.array(country_counts).reshape(1, len(country_counts)), columns=countries)

    devices = bids_data['device'].unique()
    device_counts = []
    for device in devices:
        device_counts.append(
            len(bids_data[bids_data['device'] == device]['bidder_id'].unique()))
    df_device = pd.DataFrame(
        np.array(device_counts).reshape(1, len(device_counts)), columns=devices)

    devices = bids_data['device'].unique()
    device_counts = []
    for device in devices:
        device_counts.append(
            len(bids_data[bids_data['device'] == device]['bidder_id'].unique()))
    df_device = pd.DataFrame(
        np.array(device_counts).reshape(1, len(device_counts)), columns=devices)

    ips = bids_data['ip'].unique()
    ip_counts = []
    for ip in ips:
        ip_counts.append(
            len(bids_data[bids_data['ip'] == ip]['bidder_id'].unique()))
    df_ip = pd.DataFrame(
        np.array(ip_counts).reshape(1, len(ip_counts)), columns=ips)

    urls = bids_data['url'].unique()
    url_counts = []
    for url in urls:
        url_counts.append(
            len(bids_data[bids_data['url'] == url]['bidder_id'].unique()))
    df_url = pd.DataFrame(
        np.array(url_counts).reshape(1, len(url_counts)), columns=urls)

    set_trace()
    bayes_counts = pd.concat([df_merchandise, df_country, df_device, df_ip, df_url], axis=1)
    
    # bidder_ids = bids_data['bidder_id'].unique()
    # num_bidders = len(bidder_ids)

    # bidders_info = []
    # # for each bidder
    # for i in range(num_bidders):
    #     if i % 10 == 0:
    #         print "%d/%d" % (i, num_bidders)
    #     # get bids by this bidder
    #     bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

    #     # number of bids by this bidder
    #     num_bids = len(bids)
    #     # number of auction by this bidder
    #     num_aucs = len(bids['auction'].unique())
    #     # number of merchandises by this bidder
    #     num_merchs = len(bids['merchandise'].unique())
    #     # number of devices used by this bidder
    #     num_devices = len(bids['device'].unique())
    #     # number of countries by this bidder
    #     num_countries = len(bids['country'].unique())
    #     # number of ips by this bidder
    #     num_ips = len(bids['ip'].unique())
    #     # number of urls by this bidder
    #     num_urls = len(bids['url'].unique())

    #     # primary merchandise. NOTE: every bidder only has one merchandise
    #     merch = bids['merchandise'].unique()[0]

    #     tmp_info = np.array([num_bids, num_aucs, num_merchs,
    #                          num_devices, num_countries, num_ips, num_urls,
    #                          merch])

    #     tmp_info = tmp_info.reshape(1, len(tmp_info))

    #     bidders_info.append(pd.DataFrame(
    #         tmp_info,  index=[bidder_ids[i]],
    #         columns=['num_bids', 'num_aucs', 'num_merchs', 'num_devices',
    #                  'num_countries', 'num_ips', 'num_urls', 'merchandise']
    #     ))

    # bidders_info = pd.concat(bidders_info, axis=0)
    # bidders_info.index.name = 'bidder_id'

    # return bidders_info


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


def gather_count_info(bids_data, item, item_list=None):
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

        # import pdb
        # pdb.set_trace()
        if item_list is None:
            # number of occurences of each unique item by this bidder
            bidders_count_info.append(bids[item].value_counts())
        else:
            item_u = bids[item].unique()
            item_u = set(item_list).intersection(item_u)
            bidders_count_info.append((bids[item].value_counts())[item_u])

        pd.concat([pd.DataFrame(bidders_count_info[0]).transpose(),
                   pd.DataFrame([bidder_ids[0]])], axis=1)

    bd_info = pd.concat(bidders_count_info, axis=1).transpose()
    bidders_count_info \
        = pd.concat([bd_info,
                     pd.DataFrame(bidder_ids, columns=['bidder_id'])], axis=1)\
        .set_index('bidder_id')

    bidders_count_info.fillna(0, inplace=True)

    return bidders_count_info


def gather_info_by_periods(bids_data):
    """
    Gather the useful infromation from bids data.
    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    bidders_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        num_periods = 0
        num_bids = np.zeros(3)
        num_aucs = np.zeros(3)
        num_devices = np.zeros(3)
        num_countries = np.zeros(3)
        num_ips = np.zeros(3)
        num_urls = np.zeros(3)
        period_info = []
        for ap in range(3):
            bids_per = bids[(bids['time'] < auction_periods[ap, 1])
                            & (bids['time'] > auction_periods[ap, 0])]

            # number of bids by this bidder
            num_bids[ap] = len(bids_per)
            # check the number of period participated
            if num_bids[ap] > 0:
                num_periods += 1

            # number of auction by this bidder
            num_aucs[ap] = len(bids_per['auction'].unique())
            # number of devices used by this bidder
            num_devices[ap] = len(bids_per['device'].unique())
            # number of countries by this bidder
            num_countries[ap] = len(bids_per['country'].unique())
            # number of ips by this bidder
            num_ips[ap] = len(bids_per['ip'].unique())
            # number of urls by this bidder
            num_urls[ap] = len(bids_per['url'].unique())

            tmp_info = np.array([num_bids[ap], num_aucs[ap],
                                 num_devices[ap], num_countries[ap],
                                 num_ips[ap], num_urls[ap]])
            tmp_info = tmp_info.reshape(1, len(tmp_info))

            period_info.append(pd.DataFrame(
                tmp_info,  index=[bidder_ids[i]],
                columns=['%d_num_bids' % ap, '%d_num_aucs' % ap,
                         '%d_num_devices' % ap, '%d_num_countries' % ap,
                         '%d_num_ips' % ap, '%d_num_urls' % ap]
            ))

        # average values for participated periods
        tmp_info = np.array([sum(num_bids), sum(num_aucs),
                             sum(num_devices), sum(num_countries),
                             sum(num_ips), sum(num_urls), num_periods * num_periods])\
            / float(num_periods)
        tmp_info = tmp_info.reshape(1, len(tmp_info))
        period_info.append(pd.DataFrame(
            tmp_info,  index=[bidder_ids[i]],
            columns=['ave_num_bids', 'ave_num_aucs',
                     'ave_num_devices', 'ave_num_countries',
                     'ave_num_ips', 'ave_num_urls', 'num_periods']
        ))
        bidders_info.append(pd.concat(period_info, axis=1))

    bidders_info = pd.concat(bidders_info, axis=0)
    bidders_info.index.name = 'bidder_id'

    bidders_info.fillna(0, inplace=True)

    return bidders_info
