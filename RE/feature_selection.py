"""
feature_selection.py

Extract the most important features.

* chi2 test:
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
"""

from sklearn.feature_selection import chi2, SelectKBest

import numpy as np
import pandas as pd

from fb_funcs import (predict_usample, append_merchandise, predict_cv,
                      fit_and_predict,
                      append_countries, keys_sig, keys_na,
                      append_bba, append_device)


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

    num_features = 10

    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
    info_test = pd.read_csv('data/info_test.csv', index_col=0)

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



    ############################################################################
    # Select features from each category
    # Analyze feature importance using ExtraTree clf
    ############################################################################
    # devices
    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
    info_test = pd.read_csv('data/info_test.csv', index_col=0)

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

    y_test_proba, y_train_proba, _\
        = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                          n_estimators=1000, p_use=None, plot_importance=True)

    devices_importance = ['num_bids', 'phone46', 'phone143', 'phone55',
                          'num_ips', 'phone110', 'num_aucs', 'num_urls',
                          'phone63', 'phone239', 'phone58', 'phone15',
                          'phone33', 'phone22', 'num_devices',
                          'phone2287', 'phone150', 'phone168',
                          'num_countries', 'phone195', 'phone2330',
                          'phone3359', 'phone1026', 'phone136', 'phone25',
                          'phone205', 'phone224', 'phone728', 'phone1030',
                          'phone65', 'phone21', 'phone28', 'phone739',
                          'phone219', 'phone90', 'phone6', 'phone469',
                          'phone640', 'phone144', 'phone996']

    devi = ['phone136', 'phone640', 'phone739', 'phone150', 'phone15',
            'phone33', 'phone1030', 'phone996', 'phone58', 'phone55',
            'phone2287', 'phone205', 'phone224', 'phone90', 'phone3359',
            'phone143', 'phone168', 'phone144', 'phone728', 'phone6',
            'phone2330', 'phone28', 'phone25', 'phone1026', 'phone21',
            'phone239', 'phone22', 'phone219', 'phone195', 'phone46', 'phone63',
            'phone65', 'phone110', 'phone469']


    ############################################################################
    # countries
    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
    info_test = pd.read_csv('data/info_test.csv', index_col=0)

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

    y_test_proba, y_train_proba, _\
        = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                          n_estimators=1000, p_use=None, plot_importance=True)

    country_importance = ['au', 'num_bids', 'num_aucs', 'sg',
                          'num_countries', 'uk', 'num_ips', 'id', 'fr',
                          'num_devices', 'za', 'num_urls', 'us', 'my',
                          'th', 'ca', 'de', 'ch', 'no', 'cn', 'in', 'ph',
                          'ar', 'lt', 'it', 'bf', 'br', 'ec', 'tw', 'nl',
                          'ua', 'lv', 'bh', 'bn', 'lu', 'ru', 'qa', 'jp',
                          'kr', 'sa']

    counti = ['ch', 'cn', 'ca', 'za', 'ec', 'ar', 'au', 'in', 'my', 'ru',
              'nl', 'no', 'tw', 'id', 'lv', 'lt', 'lu', 'th', 'fr', 'jp', 'bn',
              'de', 'bh', 'it', 'br', 'ph', 'sg', 'us', 'qa', 'kr', 'uk', 'bf',
              'sa', 'ua']


    ############################################################################
    # bids-count-by-bots information
    # countries
    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
    info_test = pd.read_csv('data/info_test.csv', index_col=0)

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

    y_test_proba, y_train_proba, _\
        = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                          n_estimators=1000, p_use=None, plot_importance=True)

    bba_importance = ['num_bids', 'bba_4', 'bba_5', 'bba_3', 'bba_1',
                      'bba_2', 'bba_6', 'num_devices', 'bba_7',
                      'num_urls', 'bba_9', 'bba_8', 'bba_12', 'bba_10',
                      'num_ips', 'bba_11', 'num_aucs', 'bba_13', 'bba_16',
                      'num_countries', 'bba_15', 'bba_14', 'bba_17',
                      'bba_21', 'bba_20', 'bba_18', 'bba_19', 'bba_29',
                      'bba_22', 'bba_25', 'bba_23', 'bba_24', 'bba_27',
                      'bba_28', 'bba_30', 'bba_26', 'bba_35', 'bba_31',
                      'bba_33', 'bba_32']

    bbai = ['bba_35', 'bba_33', 'bba_32', 'bba_31', 'bba_30', 'bba_19',
            'bba_18', 'bba_15', 'bba_14', 'bba_17', 'bba_16', 'bba_11', 'bba_10',
            'bba_13', 'bba_12', 'bba_28', 'bba_29', 'bba_20', 'bba_21', 'bba_22',
            'bba_23', 'bba_24', 'bba_25', 'bba_26', 'bba_27', 'bba_9', 'bba_8',
            'bba_5', 'bba_4', 'bba_7', 'bba_6', 'bba_1', 'bba_3', 'bba_2']

    ############################################################################
    # Merchandise information
    # countries
    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
    info_test = pd.read_csv('data/info_test.csv', index_col=0)

    # Count data + Merchandise data
    info_humans = append_merchandise(info_humans, drop=True)
    info_bots = append_merchandise(info_bots, drop=True)
    info_test = append_merchandise(info_test, drop=True)

    y_test_proba, y_train_proba, _\
        = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                          n_estimators=1000, p_use=None, plot_importance=True)

    merchandise_importance = ['num_bids', 'num_aucs', 'num_ips',
                              'num_urls', 'num_devices', 'num_countries',
                              'mobile', 'jewelry', 'sporting goods',
                              'office equipment', 'home goods',
                              'computers', 'books and music', 'furniture',
                              'clothing', 'auto parts', 'num_merchs']

    ci = ['computers', 'office equipment', 'auto parts', 'sporting goods',
     'books and music', 'clothing', 'furniture', 'jewelry', 'mobile',
     'home goods']



    ############################################################################
    # basic count information
    info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
    info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
    info_test = pd.read_csv('data/info_test.csv', index_col=0)
    info_humans.drop('merchandise', inplace=True, axis=1)
    info_bots.drop('merchandise', inplace=True, axis=1)
    info_test.drop('merchandise', inplace=True, axis=1)

    y_test_proba, y_train_proba, _\
        = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                          n_estimators=1000, p_use=None, plot_importance=True)

    count_importance = ['num_bids', 'num_aucs', 'num_ips', 'num_devices',
                        'num_urls', 'num_countries', 'num_merchs']
