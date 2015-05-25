"""
utils.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""


import numpy as np
import pandas as pd


merchandise_keys = ['auto parts',
                    'books and music',
                    'clothing',
                    'computers',
                    'furniture',
                    'home goods',
                    'jewelry',
                    'mobile',
                    'office equipment',
                    'sporting goods']

# countries that can be significant: derived in analysis.py
keys_sig = ['ar', 'au', 'bd', 'dj', 'ga', 'gq', 'id', 'mc', 'ml',
            'mr', 'mz', 'nl', 'th']
keys_na = ['an', 'aw', 'bi', 'cf', 'er', 'gi', 'gn', 'gp', 'mh', 'nc',
           'sb', 'tc', 'vi', 'ws']


def append_merchandises(info, drop=True):
    """
    Append merchandises a dummy variable and drop if needed.
    """

    merchandise = info.merchandise
    merchandise = pd.get_dummies(merchandise)
    for key in merchandise_keys:
        if key not in merchandise.keys():
            print key, "added."
            merchandise[key] \
                = pd.Series(np.zeros(len(merchandise)),
                            index=merchandise.index)
    info = pd.concat([info, merchandise], axis=1)

    if drop == True:
        info.drop('merchandise', axis=1, inplace=True)

    return info


def append_bba(info, bbainfo, num_bba):
    """
    Append bids-by-auction data
    """

    return pd.concat([info, bbainfo.iloc[:, :num_bba]], axis=1)


def append_countries(info, cinfo, countries):
    """
    Append country counts

    info: bidder info
    cinfo: bidder country info
    countries: countries to be appended
    """

    countries_appended = []
    for key in countries:
        # for key in list(cinfo.keys()):
        if key in list(cinfo.keys()):
            countries_appended.append(cinfo[key])
        else:
            # just create zero-column
            countries_appended.append(
                pd.DataFrame(np.zeros(len(cinfo)), index=cinfo.index, columns=[key]))

    countries_appended.append(info)

    info = pd.concat(countries_appended, axis=1)

    return info


def append_devices(info, dinfo, devices):
    """
    Append device counts

    info: bidder info
    dinfo: bidder country info
    countries: countries to be appended
    """

    devices_appended = []
    for key in devices:
        # for key in list(dinfo.keys()):
        if key in list(dinfo.keys()):
            devices_appended.append(dinfo[key])
        else:
            # just create zero-column
            devices_appended.append(
                pd.DataFrame(np.zeros(len(dinfo)), index=dinfo.index, columns=[key]))

    devices_appended.append(info)

    info = pd.concat(devices_appended, axis=1)

    return info


def append_bids_intervals(info, biinfo, bids_intervals):
    """
    Append bids interval data

    info: bidder info
    biinfo: bidder country info
    bids_intervals: bids_intervals to be appended
    """

    bids_intervals_appended = []
    for key in bids_intervals:
        if key in list(biinfo.keys()):
            bids_intervals_appended.append(biinfo[key])
        else:
            # just create zero-column
            bids_intervals_appended.append(
                pd.DataFrame(np.zeros(len(biinfo)),
                             index=biinfo.index, columns=[key]))

    bids_intervals_appended.append(info)

    info = pd.concat(bids_intervals_appended, axis=1)

    return info


def append_info(info, info_new, keys_appended):
    """
    Append info
    """

    info_appended = []
    for key in keys_appended:
        if key in list(info_new.keys()):
            info_appended.append(info_new[key])
        else:
            # just create zero-column
            info_appended.append(
                pd.DataFrame(np.zeros(len(info_new)),
                             index=info_new.index, columns=[key]))

    info_appended.append(info)

    info = pd.concat(info_appended, axis=1)

    return info
