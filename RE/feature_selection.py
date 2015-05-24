"""
feature_selection.py

Extract the most important features.

* chi2 test:
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
"""

from sys import argv

from sklearn.feature_selection import chi2, SelectKBest

import numpy as np
import pandas as pd

from fb_funcs import (append_merchandise, predict_cv,
                      fit_and_predict,
                      append_countries, keys_sig, keys_na,
                      append_bba, append_device, append_bids_intervals,
                      append_info)


def select_k_best_features(num_features, info_humans, info_bots):

    num_humans = info_humans.shape[0]
    num_bots = info_bots.shape[0]

    pd.concat([info_humans, info_bots], axis=0)

    info_train = pd.concat([info_humans, info_bots], axis=0)
    info_train.fillna(0, inplace=True)

    y_train = np.hstack([np.zeros(num_humans), np.ones(num_bots)])
    X_train = info_train.as_matrix()

    # chi2 test and select k best features
    ch2 = SelectKBest(chi2, k=num_features)
    X_train = ch2.fit_transform(X_train, y_train)

    indices_extracted = ch2.get_support(indices=True)
    features_extracted = list(info_train.keys()[indices_extracted])

    return indices_extracted, features_extracted

# num_features = 10

############################################################################
# Select features from each category
if __name__ == '__main__':

    params = {'n_estimators': 3000, 'max_features': 0.01,
              'criterion': 'gini', 'plot_importance': False, 'verbose': 1}

    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
    info_test = pd.read_csv('data/info_test.csv', index_col=0)
    
    if 'all' in argv[1]:
        num_features = 10

        # device info
        dinfo_humans = pd.read_csv('data/device_info_humans.csv', index_col=0)
        dinfo_bots = pd.read_csv('data/device_info_bots.csv', index_col=0)
        dindx_ex, dft_ex = select_k_best_features(
            num_features, dinfo_humans, dinfo_bots)

        # country info
        cinfo_humans = pd.read_csv('data/country_info_humans.csv', index_col=0)
        cinfo_bots = pd.read_csv('data/country_info_bots.csv', index_col=0)
        cindx_ex, cft_ex = select_k_best_features(
            num_features, cinfo_humans, cinfo_bots)

        # bids-by-auction info
        bbainfo_humans = pd.read_csv('data/bba_info_humans.csv', index_col=0)
        bbainfo_bots = pd.read_csv('data/bba_info_bots.csv', index_col=0)
        bbaindx_ex, bbaft_ex = select_k_best_features(
            num_features, bbainfo_humans, bbainfo_bots)

        # Count data + Merchandise data
        info_humans = append_merchandise(info_humans, drop=True)
        info_bots = append_merchandise(info_bots, drop=True)
        indx_ex, ft_ex = select_k_best_features(
            num_features, info_humans, info_bots)

        # bids interval info
        biinfo_humans = pd.read_csv('data/bids_intervals_info_humans.csv', index_col=0)
        biinfo_bots = pd.read_csv('data/bids_intervals_info_bots.csv', index_col=0)
        biindx_ex, bift_ex = select_k_best_features(
            num_features, biinfo_humans, biinfo_bots)

        # sametime bids info
        nbsinfo_humans = pd.read_csv('data/num_bids_sametime_info_humans.csv',
                                     index_col=0)
        nbsinfo_bots = pd.read_csv('data/num_bids_sametime_info_bots.csv',
                                   index_col=0)
        nbsindx_ex, nbsft_ex = select_k_best_features(
            min(num_features, nbsinfo_bots.shape[1]), nbsinfo_humans, nbsinfo_bots)

        # bid streak info
        bstrinfo_humans = pd.read_csv('data/bid_streaks_info_humans.csv', index_col=0)
        bstrinfo_bots = pd.read_csv('data/bid_streaks_info_bots.csv', index_col=0)
        bstrindx_ex, bstrft_ex = select_k_best_features(
            num_features, bstrinfo_humans, bstrinfo_bots)

        # url info
        urlinfo_humans = pd.read_csv('data/url_info_humans.csv', index_col=0)
        urlinfo_bots = pd.read_csv('data/url_info_bots.csv', index_col=0)
        urlindx_ex, urlft_ex = select_k_best_features(
            num_features, urlinfo_humans, urlinfo_bots)

        # bids by each period info
        bcepinfo_humans = pd.read_csv('data/info_humans_bp.csv', index_col=0)
        bcepinfo_bots = pd.read_csv('data/info_bots_bp.csv', index_col=0)
        bcepindx_ex, bcepft_ex = select_k_best_features(
            num_features, bcepinfo_humans, bcepinfo_bots)

        print dft_ex+cft_ex+bbaft_ex+ft_ex+bift_ex+nbsft_ex+bstrft_ex+urlft_ex+bcepft_ex
        chi2_importance = ['phone115', 'phone119', 'phone122',
                           'phone13', 'phone17', 'phone237',
                           'phone389', 'phone46', 'phone62',
                           'phone718', 'at', 'au', 'ca', 'de', 'in',
                           'jp', 'kr', 'ru', 'th', 'us', 'bba_1',
                           'bba_14', 'bba_2', 'bba_3', 'bba_4',
                           'bba_5', 'bba_6', 'bba_7', 'bba_8',
                           'bba_9', 'computers', 'jewelry', 'mobile',
                           'num_aucs', 'num_bids', 'num_countries',
                           'num_devices', 'num_ips', 'num_urls',
                           'sporting goods', 'int_1', 'int_2',
                           'int_3', 'int_4', 'int_5', 'int_6',
                           'int_7', 'int_8', 'int_9', 'int_10',
                           'num_bids_sametime_sameauc',
                           'num_bids_sametime_diffauc',
                           'num_bids_sametime', 'streak_0',
                           'streak_1', 'streak_2', 'streak_3',
                           'streak_4', 'streak_5', 'streak_16',
                           'streak_17', 'streak_18', 'streak_19',
                           '1oca0jddhorxegc', '4dd8ei0o5oqsua3',
                           '8zdkeqk4yby6lz2', 'dfq5jruldorlp4s',
                           'h5wdfy986krhq09', 'lacduz3i6mjlfkd',
                           'n7hs0kmoakimcyr', 'vasstdc27m7nks3',
                           'wk7fmlk1y5f4o18', 'zjz14bizijhg15h',
                           '0_num_bids', '0_num_ips', '1_num_bids',
                           '1_num_ips', '2_num_bids', '2_num_ips',
                           '2_num_urls', 'ave_num_bids',
                           'ave_num_ips', 'ave_num_urls']

        
    ##########################################################################
    # Select features from each category
    # Analyze feature importance using ExtraTree clf
    ##########################################################################

    elif 'device' in argv[1]:
        # device info
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

        info_humans.drop('merchandise', inplace=True, axis=1)
        info_bots.drop('merchandise', inplace=True, axis=1)
        info_test.drop('merchandise', inplace=True, axis=1)

        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        device_importance = ['phone46', 'num_bids', 'phone143',
                             'num_ips', 'num_aucs', 'phone55',
                             'num_urls', 'phone63', 'phone2287',
                             'phone2330', 'phone239', 'phone110',
                             'phone3359', 'phone168', 'num_devices',
                             'phone22', 'num_countries', 'phone33',
                             'phone205', 'phone150', 'phone1026',
                             'phone728', 'phone136', 'phone25',
                             'phone224', 'phone640', 'phone1166',
                             'phone892', 'phone2955', 'phone1013',
                             'phone195', 'phone58', 'phone4479',
                             'phone469', 'phone90', 'phone15',
                             'phone996', 'phone5479', 'phone792',
                             'phone4']

    ##########################################################################
    elif 'countr' in argv[1]:
        # country info
        cinfo_humans = pd.read_csv('data/country_info_humans.csv', index_col=0)
        cinfo_bots = pd.read_csv('data/country_info_bots.csv', index_col=0)
        cinfo_test = pd.read_csv('data/country_info_test.csv', index_col=0)

        country_appended = cinfo_humans.keys()\
                                       .union(cinfo_bots.keys())\
                                       .union(cinfo_test.keys())
        info_humans = append_countries(info_humans, cinfo_humans, country_appended)
        info_bots = append_countries(info_bots, cinfo_bots, country_appended)
        info_test = append_countries(info_test, cinfo_test, country_appended)

        info_humans.fillna(0, inplace=True)
        info_bots.fillna(0, inplace=True)
        info_test.fillna(0, inplace=True)

        info_humans.drop('merchandise', inplace=True, axis=1)
        info_bots.drop('merchandise', inplace=True, axis=1)
        info_test.drop('merchandise', inplace=True, axis=1)

        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        country_importance = ['au', 'num_bids', 'num_ips', 'sg', 'za',
                              'num_aucs', 'uk', 'fr', 'num_urls',
                              'num_devices', 'de', 'th',
                              'num_countries', 'ch', 'us', 'my', 'id',
                              'ca', 'no', 'cn']

    ##########################################################################
    # bids by aucs information
    elif 'bids-by-aucs' in argv[1]:
        # bids-by-auction info
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

        info_humans.drop('merchandise', inplace=True, axis=1)
        info_bots.drop('merchandise', inplace=True, axis=1)
        info_test.drop('merchandise', inplace=True, axis=1)

        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        bba_importance = ['num_bids', 'bba_4', 'bba_1', 'bba_5',
                          'bba_3', 'bba_2', 'bba_6', 'num_urls',
                          'num_devices', 'num_ips', 'bba_9', 'bba_7',
                          'bba_8', 'bba_10', 'bba_11', 'bba_12',
                          'bba_13', 'bba_16', 'bba_14', 'num_aucs',
                          'bba_15', 'num_countries', 'bba_17',
                          'bba_18', 'bba_19', 'bba_20', 'bba_21',
                          'bba_22', 'bba_23', 'bba_24', 'bba_25',
                          'bba_29', 'bba_27', 'bba_26', 'bba_28',
                          'bba_30', 'bba_31', 'bba_35', 'bba_36',
                          'bba_32']

    ##########################################################################
    # Merchandise information
    elif 'merchandise' in argv[1]:
        # Count data + Merchandise data
        info_humans = append_merchandise(info_humans, drop=True)
        info_bots = append_merchandise(info_bots, drop=True)
        info_test = append_merchandise(info_test, drop=True)

        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        merchandise_importance = ['num_bids', 'num_ips', 'num_urls',
                                  'num_aucs', 'num_devices',
                                  'num_countries', 'sporting goods',
                                  'mobile', 'computers', 'jewelry',
                                  'home goods', 'office equipment',
                                  'books and music', 'furniture',
                                  'auto parts', 'clothing']

    ############################################################################
    # Bids interval data
    elif 'bids_interval' in argv[1]:
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

        info_humans.drop('merchandise', inplace=True, axis=1)
        info_bots.drop('merchandise', inplace=True, axis=1)
        info_test.drop('merchandise', inplace=True, axis=1)
        
        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        bi_importance = ['int_2', 'int_8', 'int_1', 'int_4', 'int_3',
                         'int_7', 'int_6', 'int_9', 'int_10',
                         'num_bids', 'int_58', 'int_5', 'int_0',
                         'int_11', 'num_ips', 'int_12', 'int_19',
                         'int_18', 'num_urls', 'num_devices',
                         'int_20', 'int_21', 'int_13', 'int_22',
                         'num_aucs', 'int_15', 'int_17',
                         'num_countries', 'int_16', 'int_27',
                         'int_46', 'int_14', 'int_26', 'int_25',
                         'int_30', 'int_32', 'int_24', 'int_36',
                         'int_29', 'int_35']
        
    ############################################################################
    # Same time bids data
    elif 'sametime' in argv[1]:
        print "Adding same-time-bids"
        nbsinfo_humans = pd.read_csv('data/num_bids_sametime_info_humans.csv',
                                     index_col=0)
        nbsinfo_bots = pd.read_csv('data/num_bids_sametime_info_humans.csv',
                                   index_col=0)
        nbsinfo_test = pd.read_csv('data/num_bids_sametime_info_humans.csv',
                                   index_col=0)

        sametime_bids_appended = nbsinfo_humans.keys()\
                                               .union(nbsinfo_bots.keys())\
                                               .union(nbsinfo_test.keys())
        info_humans = append_info(info_humans, nbsinfo_humans,
                                  sametime_bids_appended)
        info_bots = append_info(info_bots, nbsinfo_bots,
                                sametime_bids_appended)
        info_test = append_info(info_test, nbsinfo_test,
                                sametime_bids_appended)

        info_humans.fillna(0, inplace=True)
        info_bots.fillna(0, inplace=True)
        info_test.fillna(0, inplace=True)

        info_humans.drop(['merchandise', 'num_merchs'], inplace=True, axis=1)
        info_bots.drop(['merchandise', 'num_merchs'], inplace=True, axis=1)
        info_test.drop(['merchandise', 'num_merchs'], inplace=True, axis=1)
        
        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        sametime_importance = ['num_countries', 'num_aucs',
                               'num_devices', 'num_urls', 'num_ips', 'num_bids',
                               'num_bids_sametime', 'num_bids_sametime_diffauc',
                               'num_bids_sametime_sameauc']
    ############################################################################
    elif 'streak' in argv[1]:
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

        info_humans.drop('merchandise', inplace=True, axis=1)
        info_bots.drop('merchandise', inplace=True, axis=1)
        info_test.drop('merchandise', inplace=True, axis=1)
        
        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        streak_importance = ['num_bids', 'num_devices', 'num_ips',
                             'num_aucs', 'num_urls', 'num_countries',
                             'streak_0', 'streak_1', 'streak_4',
                             'streak_2', 'streak_6', 'streak_3',
                             'streak_5', 'streak_7', 'streak_17',
                             'streak_13', 'streak_15', 'streak_8',
                             'streak_16', 'streak_18', 'streak_10',
                             'streak_11', 'streak_9', 'streak_19',
                             'streak_14', 'streak_12', 'num_merchs']

    ############################################################################
    elif 'url' in argv[1]:
        # Bid url data
        print "Adding bid url data"
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

        info_humans.drop('merchandise', inplace=True, axis=1)
        info_bots.drop('merchandise', inplace=True, axis=1)
        info_test.drop('merchandise', inplace=True, axis=1)
        
        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        url_importance = ['num_bids', 'vasstdc27m7nks3', 'num_ips',
                          'num_urls', 'num_devices', 'num_aucs',
                          'num_countries', 'szyjr65zi6h3qbz',
                          'wuientgh43dvm2q', 'lacduz3i6mjlfkd',
                          'z1j3lnl5ph0e6nl', 's85ymwlo8uqfy7j',
                          'pkegaymari9jblo', '301o49axv6udhkl',
                          'nzho4dxbsqsy9nc', '575tu52ly8ikuqs',
                          'ihc409avhf40y1l', 'm4czoknep5wf2ff',
                          '7ajclfubja9y644', 'f7hbdb3527v9te6']

    ############################################################################
    elif 'each' in argv[1]:
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
        info_humans.drop(['merchandise', 'num_merchs'], inplace=True, axis=1)
        info_bots.drop(['merchandise', 'num_merchs'], inplace=True, axis=1)
        info_test.drop(['merchandise', 'num_merchs'], inplace=True, axis=1)
        
        y_test_proba, y_train_proba, _, features\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              params=params)

        each_importance = ['ave_num_bids', 'num_bids', 'ave_num_ips',
                           'num_ips', 'ave_num_urls', '2_num_bids',
                           'num_urls', 'ave_num_aucs', 'num_aucs',
                           'ave_num_devices', 'num_devices',
                           '0_num_bids', 'ave_num_countries',
                           'num_countries', '2_num_ips', '2_num_urls',
                           '1_num_bids', '2_num_aucs',
                           '2_num_devices', '0_num_aucs', '0_num_ips',
                           '2_num_countries', '0_num_devices',
                           '0_num_urls', '1_num_ips', '1_num_aucs',
                           '1_num_devices', '1_num_urls',
                           '0_num_countries', '1_num_countries',
                           'num_periods']
        
    ##########################################################################
    # basic count information
    elif 'basic' in argv[1]:
        info_humans.drop('merchandise', inplace=True, axis=1)
        info_bots.drop('merchandise', inplace=True, axis=1)
        info_test.drop('merchandise', inplace=True, axis=1)

        y_test_proba, y_train_proba, _\
            = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                              n_estimators=1000, p_use=None, plot_importance=True)

        count_importance = ['num_bids', 'num_aucs', 'num_ips', 'num_devices',
                            'num_urls', 'num_countries', 'num_merchs']

    else:
        # gather all the extracted features
        combined_features = ['phone115', 'phone119', 'phone122', 'phone13', 'phone17', 'phone237', 'phone389', 'phone46', 'phone62', 'phone718', 'at', 'au', 'ca', 'de', 'in', 'jp', 'kr', 'ru', 'th', 'us', 'bba_1', 'bba_14', 'bba_2', 'bba_3', 'bba_4', 'bba_5', 'bba_6', 'bba_7', 'bba_8', 'bba_9', 'computers', 'jewelry', 'mobile', 'num_aucs', 'num_bids', 'num_countries', 'num_devices', 'num_ips', 'num_urls', 'sporting goods', 'int_1', 'int_2', 'int_3', 'int_4', 'int_5', 'int_6', 'int_7', 'int_8', 'int_9', 'int_10', 'num_bids_sametime_sameauc', 'num_bids_sametime_diffauc', 'num_bids_sametime', 'streak_0', 'streak_1', 'streak_2', 'streak_3', 'streak_4', 'streak_5', 'streak_16', 'streak_17', 'streak_18', 'streak_19', '1oca0jddhorxegc', '4dd8ei0o5oqsua3', '8zdkeqk4yby6lz2', 'dfq5jruldorlp4s', 'h5wdfy986krhq09', 'lacduz3i6mjlfkd', 'n7hs0kmoakimcyr', 'vasstdc27m7nks3', 'wk7fmlk1y5f4o18', 'zjz14bizijhg15h', '0_num_bids', '0_num_ips', '1_num_bids', '1_num_ips', '2_num_bids', '2_num_ips', '2_num_urls', 'ave_num_bids', 'ave_num_ips', 'ave_num_urls']
        device_importance = ['phone46', 'num_bids', 'phone143',
                             'num_ips', 'num_aucs', 'phone55',
                             'num_urls', 'phone63', 'phone2287',
                             'phone2330', 'phone239', 'phone110',
                             'phone3359', 'phone168', 'num_devices',
                             'phone22', 'num_countries', 'phone33',
                             'phone205', 'phone150', 'phone1026',
                             'phone728', 'phone136', 'phone25',
                             'phone224', 'phone640', 'phone1166',
                             'phone892', 'phone2955', 'phone1013',
                             'phone195', 'phone58', 'phone4479',
                             'phone469', 'phone90', 'phone15',
                             'phone996', 'phone5479', 'phone792',
                             'phone4']

        country_importance = ['au', 'num_bids', 'num_ips', 'sg', 'za',
                              'num_aucs', 'uk', 'fr', 'num_urls',
                              'num_devices', 'de', 'th',
                              'num_countries', 'ch', 'us', 'my', 'id',
                              'ca', 'no', 'cn']
        bba_importance = ['num_bids', 'bba_4', 'bba_1', 'bba_5',
                          'bba_3', 'bba_2', 'bba_6', 'num_urls',
                          'num_devices', 'num_ips', 'bba_9', 'bba_7',
                          'bba_8', 'bba_10', 'bba_11', 'bba_12',
                          'bba_13', 'bba_16', 'bba_14', 'num_aucs',
                          'bba_15', 'num_countries', 'bba_17',
                          'bba_18', 'bba_19', 'bba_20', 'bba_21',
                          'bba_22', 'bba_23', 'bba_24', 'bba_25',
                          'bba_29', 'bba_27', 'bba_26', 'bba_28',
                          'bba_30', 'bba_31', 'bba_35', 'bba_36',
                          'bba_32']
        merchandise_importance = ['num_bids', 'num_ips', 'num_urls',
                                  'num_aucs', 'num_devices',
                                  'num_countries', 'sporting goods',
                                  'mobile', 'computers', 'jewelry',
                                  'home goods', 'office equipment',
                                  'books and music', 'furniture',
                                  'auto parts', 'clothing']

        bi_importance = ['int_2', 'int_8', 'int_1', 'int_4', 'int_3',
                         'int_7', 'int_6', 'int_9', 'int_10',
                         'num_bids', 'int_58', 'int_5', 'int_0',
                         'int_11', 'num_ips', 'int_12', 'int_19',
                         'int_18', 'num_urls', 'num_devices',
                         'int_20', 'int_21', 'int_13', 'int_22',
                         'num_aucs', 'int_15', 'int_17',
                         'num_countries', 'int_16', 'int_27',
                         'int_46', 'int_14', 'int_26', 'int_25',
                         'int_30', 'int_32', 'int_24', 'int_36',
                         'int_29', 'int_35']
        sametime_importance = ['num_countries', 'num_aucs',
                               'num_devices', 'num_urls', 'num_ips', 'num_bids',
                               'num_bids_sametime', 'num_bids_sametime_diffauc',
                               'num_bids_sametime_sameauc']

        streak_importance = ['num_bids', 'num_devices', 'num_ips',
                             'num_aucs', 'num_urls', 'num_countries',
                             'streak_0', 'streak_1', 'streak_4',
                             'streak_2', 'streak_6', 'streak_3',
                             'streak_5', 'streak_7', 'streak_17',
                             'streak_13', 'streak_15', 'streak_8',
                             'streak_16', 'streak_18', 'streak_10',
                             'streak_11', 'streak_9', 'streak_19',
                             'streak_14', 'streak_12', 'num_merchs']

        url_importance = ['num_bids', 'vasstdc27m7nks3', 'num_ips',
                          'num_urls', 'num_devices', 'num_aucs',
                          'num_countries', 'szyjr65zi6h3qbz',
                          'wuientgh43dvm2q', 'lacduz3i6mjlfkd',
                          'z1j3lnl5ph0e6nl', 's85ymwlo8uqfy7j',
                          'pkegaymari9jblo', '301o49axv6udhkl',
                          'nzho4dxbsqsy9nc', '575tu52ly8ikuqs',
                          'ihc409avhf40y1l', 'm4czoknep5wf2ff',
                          '7ajclfubja9y644', 'f7hbdb3527v9te6']

        each_importance = ['ave_num_bids', 'num_bids', 'ave_num_ips',
                           'num_ips', 'ave_num_urls', '2_num_bids',
                           'num_urls', 'ave_num_aucs', 'num_aucs',
                           'ave_num_devices', 'num_devices',
                           '0_num_bids', 'ave_num_countries',
                           'num_countries', '2_num_ips', '2_num_urls',
                           '1_num_bids', '2_num_aucs',
                           '2_num_devices', '0_num_aucs', '0_num_ips',
                           '2_num_countries', '0_num_devices',
                           '0_num_urls', '1_num_ips', '1_num_aucs',
                           '1_num_devices', '1_num_urls',
                           '0_num_countries', '1_num_countries',
                           'num_periods']

        features_extracted = list(set( combined_features +
                                       device_importance +
                                       country_importance +
                                       bba_importance +
                                       merchandise_importance +
                                       bi_importance +
                                       sametime_importance +
                                       streak_importance +
                                       url_importance +
                                       each_importance ))
