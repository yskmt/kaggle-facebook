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
info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
info_humans = append_merchandise(info_humans, drop=True)
info_bots = append_merchandise(info_bots, drop=True)
indx_ex, ft_ex = select_k_best_features(
    num_features, info_humans, info_bots)

