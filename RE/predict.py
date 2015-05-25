"""
predict.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

from pdb import set_trace
from sys import argv
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

import fb_funcs
from fb_funcs import (append_merchandise, predict_cv,
                      fit_and_predict, append_info,
                      append_countries, keys_sig, keys_na,
                      append_bba, append_device, append_bids_intervals)
import ffs


if len(argv)==1:
    ############################################################################
    # Load basic data
    ############################################################################
    print "Loading postprocessed data files..."

    humansfile = 'data/info_humans.csv'
    botsfile = 'data/info_bots.csv'
    testfile = 'data/info_test.csv'

    info_humans = pd.read_csv(humansfile, index_col=0)
    info_bots = pd.read_csv(botsfile, index_col=0)
    info_test = pd.read_csv(testfile, index_col=0)

    num_humans = info_humans.shape[0]
    num_bots = info_bots.shape[0]
    num_test = info_test.shape[0]


    ############################################################################
    # Data appending
    ############################################################################

    ############################################################################
    # Merchandise data
    print "Adding merchandise data..."
    info_humans = append_merchandise(info_humans, drop=True)
    info_bots = append_merchandise(info_bots, drop=True)
    info_test = append_merchandise(info_test, drop=True)

    ############################################################################
    # Country data
    print "Adding country data..."
    cinfo_humans = pd.read_csv('data/country_info_humans.csv', index_col=0)
    cinfo_bots = pd.read_csv('data/country_info_bots.csv', index_col=0)
    cinfo_test = pd.read_csv('data/country_info_test.csv', index_col=0)

    cts_appended = cinfo_humans.keys().union(cinfo_bots.keys())
    info_humans = append_countries(info_humans, cinfo_humans, cts_appended)
    info_bots = append_countries(info_bots, cinfo_bots, cts_appended)
    info_test = append_countries(info_test, cinfo_test, cts_appended)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    ############################################################################
    # Device data
    print "Adding devices data"
    dinfo_humans = pd.read_csv('data/device_info_humans.csv', index_col=0)
    dinfo_bots = pd.read_csv('data/device_info_bots.csv', index_col=0)
    dinfo_test = pd.read_csv('data/device_info_test.csv', index_col=0)

    devices_appended = dinfo_humans.keys()\
                                   .union(dinfo_bots.keys())\
                                   .union(dinfo_test.keys())
    info_humans = append_device(info_humans, dinfo_humans, devices_appended)
    info_bots = append_device(info_bots, dinfo_bots, devices_appended)
    info_test = append_device(info_test, dinfo_test, devices_appended)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    ############################################################################
    # Bids count by auction data
    print "Adding bids-count-by-auction data..."
    bbainfo_humans = pd.read_csv('data/bba_info_humans.csv', index_col=0)
    bbainfo_bots = pd.read_csv('data/bba_info_bots.csv', index_col=0)
    bbainfo_test = pd.read_csv('data/bba_info_test.csv', index_col=0)

    # take the minimum of the number of auctions
    min_bba = np.min([bbainfo_humans.shape[1],
                      bbainfo_bots.shape[1],
                      bbainfo_test.shape[1]])
    min_bba = 100

    info_humans = append_bba(info_humans, bbainfo_humans, min_bba)
    info_bots = append_bba(info_bots, bbainfo_bots, min_bba)
    info_test = append_bba(info_test, bbainfo_test, min_bba)

    ############################################################################
    # Bids interval data
    print "Adding bids interval data"
    biinfo_humans = pd.read_csv('data/bids_intervals_info_humans.csv', index_col=0)
    biinfo_bots = pd.read_csv('data/bids_intervals_info_bots.csv', index_col=0)
    biinfo_test = pd.read_csv('data/bids_intervals_info_test.csv', index_col=0)

    bids_intervals_appended = biinfo_humans.keys()\
                                           .union(biinfo_bots.keys())\
                                           .union(biinfo_test.keys())
    info_humans = append_bids_intervals(info_humans, biinfo_humans,
                                        bids_intervals_appended)
    info_bots = append_bids_intervals(info_bots, biinfo_bots,
                                      bids_intervals_appended)
    info_test = append_bids_intervals(info_test, biinfo_test,
                                      bids_intervals_appended)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    ############################################################################
    # Number of same-time-bids data
    print "Adding same-time bids data"
    nbsinfo_humans = pd.read_csv(
        'data/num_bids_sametime_info_humans.csv', index_col=0)
    nbsinfo_bots = pd.read_csv('data/num_bids_sametime_info_bots.csv', index_col=0)
    nbsinfo_test = pd.read_csv('data/num_bids_sametime_info_test.csv', index_col=0)

    keys_nbs = nbsinfo_humans.keys()
    info_humans = append_info(info_humans, nbsinfo_humans, keys_nbs)
    info_bots = append_info(info_bots, nbsinfo_bots, keys_nbs)
    info_test = append_info(info_test, nbsinfo_test, keys_nbs)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    ############################################################################
    # Bid streak data
    print "Adding bid streak data"
    bstrinfo_humans = pd.read_csv('data/bid_streaks_info_humans.csv', index_col=0)
    bstrinfo_bots = pd.read_csv('data/bid_streaks_info_bots.csv', index_col=0)
    bstrinfo_test = pd.read_csv('data/bid_streaks_info_test.csv', index_col=0)

    keys_bstr = bstrinfo_humans.keys()
    info_humans = append_info(info_humans, bstrinfo_humans, keys_bstr)
    info_bots = append_info(info_bots, bstrinfo_bots, keys_bstr)
    info_test = append_info(info_test, bstrinfo_test, keys_bstr)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)
    ############################################################################
    # url data
    print "Adding url data"
    urlinfo_humans = pd.read_csv('data/url_info_humans.csv', index_col=0)
    urlinfo_bots = pd.read_csv('data/url_info_bots.csv', index_col=0)
    urlinfo_test = pd.read_csv('data/url_info_test.csv', index_col=0)

    keys_url = urlinfo_humans.keys()
    info_humans = append_info(info_humans, urlinfo_humans, keys_url)
    info_bots = append_info(info_bots, urlinfo_bots, keys_url)
    info_test = append_info(info_test, urlinfo_test, keys_url)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    ############################################################################
    # bid counts for each period data
    print "Adding bid count for each period data"
    bcepinfo_humans = pd.read_csv('data/info_humans_bp.csv', index_col=0)
    bcepinfo_bots = pd.read_csv('data/info_bots_bp.csv', index_col=0)
    bcepinfo_test = pd.read_csv('data/info_test_bp.csv', index_col=0)

    keys_bcep = bcepinfo_humans.keys()
    info_humans = append_info(info_humans, bcepinfo_humans, keys_bcep)
    info_bots = append_info(info_bots, bcepinfo_bots, keys_bcep)
    info_test = append_info(info_test, bcepinfo_test, keys_bcep)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)


    ############################################################################
    # Outlier dropping
    ############################################################################
    print "Removing outliers..."

    bots_outliers = [
        '7fab82fa5eaea6a44eb743bc4bf356b3tarle',
        'f35082c6d72f1f1be3dd23f949db1f577t6wd',
        'bd0071b98d9479130e5c053a244fe6f1muj8h',
        '91c749114e26abdb9a4536169f9b4580huern',
        '74a35c4376559c911fdb5e9cfb78c5e4btqew'
    ]
    info_bots.drop(bots_outliers, inplace=True)


    ############################################################################
    # Feature selection
    ############################################################################
    print "Selecting features..."

    # first dropping merchandise feature...
    keys_all = info_humans.keys()
    if 'merchandise' in keys_all:
        keys_use = keys_all.drop(['merchandise'])
    else:
        keys_use = keys_all

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    keys_use = keys_all
    keys_use = ['f7hbdb3527v9te6', 'bba_19', 'bba_18', 'sporting goods', '0_num_ips', 'bba_15', 'bba_14', 'streak_5', 'streak_4', 'streak_3', 'streak_2', 'streak_1', 'streak_0', 'num_bids', 'phone640', 'bba_21', 'ihc409avhf40y1l', 'phone4479', 's85ymwlo8uqfy7j', 'phone205', 'bba_26', 'th', 'furniture', '7ajclfubja9y644', 'int_5', 'home goods', 'de', 'bba_28', 'bba_29', 'phone1166', 'streak_6', 'phone150', 'bba_20', '2_num_ips', 'bba_22', 'bba_23', 'bba_24', 'bba_25', 'phone5479', 'bba_27', 'num_devices', 'phone195', 'phone46', 'ave_num_bids', 'zjz14bizijhg15h', 'phone119', '2_num_bids', 'phone110', 'phone115', 'ave_num_aucs', 'bba_36', 'bba_35', 'bba_32', 'bba_31', 'bba_30', 'streak_8', 'za', '0_num_bids', 'vasstdc27m7nks3', 'sg', 'phone33', 'int_46', 'phone996', 'bba_17', 'ru', 'bba_16', 'phone58', 'jewelry', 'phone55', '2_num_aucs', 'phone792', 'bba_10', 'streak_14', 'bba_13', '0_num_devices', 'num_countries', 'bba_12', '0_num_urls', 'z1j3lnl5ph0e6nl', 'phone168', 'streak_13', '2_num_countries', 'jp', 'ave_num_countries', 'phone4', 'int_58', 'phone62', 'streak_7', 'streak_12', '8zdkeqk4yby6lz2', 'int_16', 'phone63', 'computers', 'lacduz3i6mjlfkd', '575tu52ly8ikuqs', 'bba_11', '301o49axv6udhkl', '1_num_ips', '0_num_aucs', 'phone469', '0_num_countries', 'phone718', '1_num_bids', 'h5wdfy986krhq09', 'ch', 'cn', 'int_9', 'int_8', 'int_7', 'int_6', 'ca', 'int_4', 'int_3', 'int_2', 'int_1', 'int_0', 'ave_num_devices', 'int_20', 'phone2955', 'phone1013', 'int_29', 'int_26', 'int_27', 'int_24', 'int_25', 'int_22', 'phone2287', 'int_21', 'n7hs0kmoakimcyr', 'phone22', 'streak_9', 'phone143', 'num_bids_sametime_diffauc', 'phone389', 'nzho4dxbsqsy9nc', 'num_periods', 'books and music', '2_num_devices', 'pkegaymari9jblo', 'int_35', '1_num_countries', 'int_36', 'ave_num_urls', 'int_30', 'us', 'int_32', 'wuientgh43dvm2q', 'dfq5jruldorlp4s', 'szyjr65zi6h3qbz', 'uk', 'streak_15', 'my', 'phone15', 'num_bids_sametime_sameauc', 'phone136', 'num_aucs', 'in', '1_num_devices', 'phone13', 'au', 'at', 'phone17', 'clothing', 'num_urls', '4dd8ei0o5oqsua3', 'ave_num_ips', 'phone892', 'no', 'streak_19', 'id', '1_num_aucs', 'int_18', 'phone224', 'phone90', 'phone237', 'phone3359', 'bba_8', 'fr', 'streak_11', 'phone122', 'num_bids_sametime', 'office equipment', 'phone728', 'auto parts', 'int_17', 'phone2330', 'int_15', 'int_14', 'int_13', 'int_12', 'int_11', 'int_10', 'phone25', 'phone1026', '1_num_urls', 'm4czoknep5wf2ff', 'phone239', 'int_19', 'wk7fmlk1y5f4o18', 'num_ips', 'streak_10', 'mobile', 'bba_1', 'kr', '2_num_urls', 'streak_18', 'bba_9', '1oca0jddhorxegc', 'bba_5', 'bba_4', 'bba_7', 'bba_6', 'streak_17', 'streak_16', 'bba_3', 'bba_2']

    keys_use = ['au', 'za', 'phone4', 'num_devices', 'uk', 'ave_num_devices', 'bba_2', 'ave_num_bids', 'th', 'bba_3', 'bba_4', 'phone63', 'phone25', 'sg', 'id', 'phone22', 'int_1', 'bba_5', 'num_ips', 'my', 'de', 'num_bids', 'us', 'num_urls', 'bba_1', 'int_8', 'phone15', 'ave_num_urls', 'ave_num_ips', 'streak_15', 'int_4', 'ca', 'int_2', 'streak_13', 'int_7', 'streak_8', 'num_aucs', 'num_countries', 'streak_17', 'in', 'phone33', 'bba_6', 'ave_num_aucs', 'int_46', 'phone224', 'ave_num_countries', 'phone55', 'streak_16', 'fr', 'int_58', 'streak_4', 'cn', 'streak_12', 'bba_9', 'phone150', 'bba_8', 'jp', 'phone195', 'int_18', 'phone46', 'bba_7', 'no', 'streak_2', 'streak_1', 'streak_14', 'streak_3', 'streak_0', 'int_19', 'int_10', 'streak_9', 'phone110', 'int_3', 'int_15', 'bba_10', 'ru', '2_num_bids', 'streak_7', 'phone17', '2_num_urls', 'streak_11', 'phone792', 'phone143', 'streak_6', 'bba_11', 'streak_10', 'num_bids_sametime_diffauc', 'streak_5', '1_num_urls', 'ch', 'phone168']

    # 0.959
    keys_use = ['au', 'phone4', 'za', 'num_devices', 'ave_num_bids', 'uk', 'bba_2', 'ave_num_devices', 'th', 'phone63', 'bba_4', 'bba_3', 'sg', 'id', 'phone25', 'num_bids', 'bba_5', 'num_ips', 'phone22', 'num_urls', 'my', 'int_8', 'ave_num_urls', 'int_1', 'us', 'bba_1', 'ave_num_aucs', 'ave_num_ips', 'num_aucs', 'de', 'int_46', 'num_countries', 'ca', 'int_7', 'phone224', 'bba_6', 'phone15', 'phone55', 'int_4', 'ave_num_countries', 'in', 'int_2', 'int_58', 'phone46', '1_num_urls', 'bba_9', 'phone33', '2_num_bids', 'bba_7', 'cn', 'phone195', 'phone150', 'int_19', 'streak_13', 'bba_8', 'fr', 'streak_8', 'streak_15', 'streak_17', 'int_10', 'bba_10', 'int_18', 'streak_14', '2_num_urls', 'streak_16', 'streak_1', 'jp', 'phone110', 'bba_11', 'phone168', 'int_15', 'no', 'phone792', 'int_3', 'streak_0', 'streak_3', 'streak_2', 'ru']

    # 0.955199083425 +- 0.0196996639719
    keys_use = ['au', 'num_devices', 'za', 'phone4', 'ave_num_bids', 'uk', 'bba_2', 'ave_num_devices', 'bba_4', 'th', 'bba_3', 'phone63', 'id', 'num_urls', 'num_bids', 'num_ips', 'bba_5', 'int_8', 'int_1', 'sg', 'phone22', 'phone25', 'ave_num_urls', 'bba_1', 'my', 'us', 'ave_num_ips', 'ave_num_aucs', 'de', 'int_7', 'num_aucs', 'int_2', 'num_countries', 'ca', 'int_4', 'phone15', 'in', 'bba_9', 'bba_6', 'phone224', 'streak_8', 'ave_num_countries', 'bba_7', 'phone46', 'phone55', 'phone33', 'int_10', 'streak_3', 'phone195', 'bba_10', 'phone150', 'streak_1', '2_num_bids', 'bba_8', '1_num_urls', 'cn', '2_num_urls', 'fr', 'streak_0', 'phone110', 'bba_11', 'streak_2', 'int_3', 'phone168', 'jp', 'phone792', 'no', 'ru', 'int_0']

    # 0.95573397572 +- 0.0173075834306
    keys_use = ['au', 'za', 'num_devices', 'phone4', 'bba_2', 'ave_num_bids', 'bba_4', 'ave_num_devices', 'uk', 'th', 'int_8', 'num_bids', 'bba_3', 'id', 'int_1', 'sg', 'bba_1', 'bba_5', 'num_ips', 'num_urls', 'ave_num_urls', 'phone22', 'int_7', 'my', 'phone25', 'streak_8', 'phone63', 'int_2', 'ave_num_ips', 'int_4', 'us', 'de', 'bba_7', 'num_aucs', 'ave_num_aucs', 'bba_6', 'streak_3', 'bba_9', 'phone15', 'streak_2', 'streak_1', 'in', 'phone46', 'ca', 'phone195', 'phone55', 'int_10', 'num_countries', '2_num_bids', 'ave_num_countries', 'bba_8', 'streak_0', 'phone224', 'bba_10', 'phone150', 'cn', 'phone33', '1_num_urls', 'int_3', '2_num_urls']

    # 0.954217649208 +- 0.0249995660696
    keys_use = ['au', 'ave_num_bids', 'za', 'phone4', 'bba_2', 'uk', 'ave_num_devices', 'th', 'bba_4', 'bba_3', 'int_8', 'id', 'bba_1', 'bba_5', 'int_1', 'my', 'sg', 'ave_num_urls', 'phone25', 'int_7', 'int_2', 'streak_8', 'phone63', 'us', 'int_4', 'de', 'phone22', 'ave_num_ips', 'bba_6', 'streak_1', 'phone46', 'ca', 'in', 'bba_9', 'ave_num_aucs', 'phone15', 'bba_7', 'int_10', 'streak_3', 'streak_2', 'phone55', 'phone195', 'bba_8', 'streak_0', 'phone224', 'ave_num_countries', 'phone150', 'cn', 'bba_10', 'int_3', 'phone33']

    # 0.954465847692 +- 0.0209881334398    
    keys_use = ['au', 'ave_num_bids', 'phone4', 'za', 'ave_num_devices', 'bba_2', 'uk', 'th', 'bba_4', 'int_8', 'id', 'bba_3', 'ave_num_urls', 'int_1', 'bba_1', 'phone22', 'sg', 'bba_5', 'phone25', 'ave_num_ips', 'int_7', 'phone63', 'my', 'int_2', 'ave_num_aucs', 'us', 'streak_8', 'de', 'int_4', 'ave_num_countries', 'ca', 'phone46', 'streak_1', 'bba_7', 'phone15', 'phone195', 'in', 'bba_6', 'bba_9', 'phone150', 'phone55', 'phone224']

    # 0.953448562872 +- 0.0224014836777
    keys_use = ['au', 'ave_num_bids', 'za', 'phone4', 'ave_num_devices', 'bba_2', 'uk', 'bba_4', 'int_8', 'bba_3', 'int_1', 'ave_num_urls', 'bba_1', 'th', 'id', 'bba_5', 'phone22', 'sg', 'ave_num_ips', 'int_2', 'int_7', 'phone25', 'streak_8', 'int_4', 'my', 'ave_num_aucs', 'streak_1', 'phone63', 'us', 'bba_9', 'bba_7', 'bba_6', 'ave_num_countries']

    # 0.954711017166 +- 0.0250041777182
    keys_use = ['au', 'phone4', 'ave_num_bids', 'ave_num_devices', 'za', 'bba_2', 'uk', 'ave_num_urls', 'bba_4', 'int_8', 'id', 'phone22', 'bba_1', 'bba_3', 'th', 'ave_num_ips', 'int_1', 'phone25', 'sg', 'bba_5', 'my', 'phone63', 'int_7', 'us', 'ave_num_aucs', 'int_2', 'int_4', 'bba_9', 'ave_num_countries']

    # 0.955017361678 +- 0.0163165216481
    keys_use = ['au', 'phone4', 'ave_num_bids', 'ave_num_devices', 'za', 'bba_2', 'uk', 'bba_4', 'ave_num_urls', 'int_8', 'bba_1', 'int_1', 'id', 'bba_3', 'ave_num_ips', 'phone22', 'bba_5', 'int_7', 'th', 'sg', 'int_2', 'phone25', 'ave_num_aucs']

    # 0.953708841758 +- 0.017063192807
    keys_use = ['au', 'phone4', 'ave_num_bids', 'ave_num_devices', 'za', 'bba_2', 'uk', 'bba_4', 'ave_num_urls', 'id', 'bba_1', 'bba_3', 'int_8', 'bba_5', 'int_1', 'ave_num_ips', 'phone22', 'th', 'sg']

    # 0.951517793864 +- 0.021983766701
    keys_use = ['au', 'phone4', 'ave_num_bids', 'ave_num_devices', 'za', 'bba_2', 'uk', 'bba_4', 'ave_num_urls', 'id', 'bba_1', 'bba_3', 'int_8', 'bba_5']


    print "Extracting keys..."
    info_humans = info_humans[keys_use]
    info_bots = info_bots[keys_use]
    info_test = info_test[keys_use]

    print "Saving prprocessed data.."
    info_humans.to_csv('data/info_humans_pp11.csv')
    info_bots.to_csv('data/info_bots_pp11.csv')
    info_test.to_csv('data/info_test_pp11.csv')

    # print "Saving prprocessed data.."
    # info_humans.to_csv('data/info_humans_all.csv')
    # info_bots.to_csv('data/info_bots_all.csv')
    # info_test.to_csv('data/info_test_all.csv')

elif 'resume' in argv[1]:
    ############################################################################
    # Save/Load preprocessed data
    ############################################################################

    print "loading preprocessed data..."
    info_humans = pd.read_csv('data/info_humans_pp9.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots_pp9.csv', index_col=0)
    info_test = pd.read_csv('data/info_test_pp9.csv', index_col=0)

else:
    sys.exit(1)
    
############################################################################
# k-fold Cross Validaton
############################################################################
# params for xgb
params = {'model': 'XGB', 'colsample_bytree': 0.367, 'silent': 1,
          'num_rounds': 1000, 'nthread': 8, 'min_child_weight': 3.0,
          'subsample': 0.9, 'eta': 0.002, 'max_depth': 5.0, 'gamma': 1.0}

# params for et
# params = {'model': 'ET', 'n_estimators': 3000, 'max_features': 'auto',
#           'criterion': 'gini', 'plot_importance': False, 'verbose': 1,
#           'n_jobs': 2}

# params for logistic regression
# params = {'model': 'logistic'}

# params for svc
# params = {'model': 'SVC', 'n_estimators': 3000, 'max_features': 'auto',
#           'criterion': 'gini', 'plot_importance': False, 'verbose': 1,
#           'n_jobs': 2}

# params for RF
# params = {'model': 'RF', 'n_estimators': 1000, 'max_features': 'auto',
#           'criterion': 'gini', 'plot_importance': False, 'verbose': 1,
#           'n_jobs': -1, 'max_depth': 3}

roc_auc_mean, roc_auc_std \
    = ffs.kfcv(info_humans, info_bots, params, num_cv=5, num_folds=5)

print "cross validation results:"
print roc_auc_mean,"+-", roc_auc_std

############################################################################
# fit and predict
############################################################################

y_test_proba, y_train_proba, cvr, feature_importance\
    = fb_funcs.fit_and_predict(info_humans, info_bots, info_test,
                               params=params)

try:
    auc_max = np.max(np.array(map(lambda x: float(x.split('\t')[1].split(':')[1].split('+')[0]), cvr)))
    ind_max = np.argmax(np.array(map(lambda x: float(x.split('\t')[1].split(':')[1].split('+')[0]), cvr)))

    print ind_max, auc_max
except:
    pass
############################################################################
# submission file generation
############################################################################

# 70 bidders in test.csv do not have any data in bids.csv. Thus they
# are not included in analysis/prediction, but they need to be
# appended in the submission. The prediction of these bidders do not matter.

print "writing a submission file..."
submission = pd.DataFrame(
    y_test_proba, index=info_test.index, columns=['prediction'])
test_bidders = pd.read_csv('data/test.csv', index_col=0)

submission = pd.concat([submission, test_bidders], axis=1)
submission.fillna(0, inplace=True)
submission.to_csv('data/submission.csv', columns=['prediction'],
                  index_label='bidder_id')
