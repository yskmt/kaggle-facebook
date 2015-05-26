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
from fb_funcs import predict_cv, fit_and_predict
from utils import (append_merchandises, append_countries, append_bba,
                   append_devices, append_bids_intervals, append_info,
                   write_submission)

import ffs


if len(argv) == 1:
    ##########################################################################
    # Load basic data
    ##########################################################################
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

    ##########################################################################
    # Data appending
    ##########################################################################

    ##########################################################################
    # Merchandise data
    print "Adding merchandise data..."
    info_humans = append_merchandises(info_humans, drop=True)
    info_bots = append_merchandises(info_bots, drop=True)
    info_test = append_merchandises(info_test, drop=True)

    ##########################################################################
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

    ##########################################################################
    # Device data
    print "Adding devices data"
    dinfo_humans = pd.read_csv('data/device_info_humans.csv', index_col=0)
    dinfo_bots = pd.read_csv('data/device_info_bots.csv', index_col=0)
    dinfo_test = pd.read_csv('data/device_info_test.csv', index_col=0)

    devices_appended = dinfo_humans.keys()\
                                   .union(dinfo_bots.keys())\
                                   .union(dinfo_test.keys())
    info_humans = append_devices(info_humans, dinfo_humans, devices_appended)
    info_bots = append_devices(info_bots, dinfo_bots, devices_appended)
    info_test = append_devices(info_test, dinfo_test, devices_appended)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    ##########################################################################
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

    ##########################################################################
    # Bids interval data
    print "Adding bids interval data"
    biinfo_humans = pd.read_csv(
        'data/bids_intervals_info_humans.csv', index_col=0)
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

    ##########################################################################
    # Number of same-time-bids data
    print "Adding same-time bids data"
    nbsinfo_humans = pd.read_csv(
        'data/num_bids_sametime_info_humans.csv', index_col=0)
    nbsinfo_bots = pd.read_csv(
        'data/num_bids_sametime_info_bots.csv', index_col=0)
    nbsinfo_test = pd.read_csv(
        'data/num_bids_sametime_info_test.csv', index_col=0)

    keys_nbs = nbsinfo_humans.keys()
    info_humans = append_info(info_humans, nbsinfo_humans, keys_nbs)
    info_bots = append_info(info_bots, nbsinfo_bots, keys_nbs)
    info_test = append_info(info_test, nbsinfo_test, keys_nbs)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)

    ##########################################################################
    # Bid streak data
    print "Adding bid streak data"
    bstrinfo_humans = pd.read_csv(
        'data/bid_streaks_info_humans.csv', index_col=0)
    bstrinfo_bots = pd.read_csv('data/bid_streaks_info_bots.csv', index_col=0)
    bstrinfo_test = pd.read_csv('data/bid_streaks_info_test.csv', index_col=0)

    keys_bstr = bstrinfo_humans.keys()
    info_humans = append_info(info_humans, bstrinfo_humans, keys_bstr)
    info_bots = append_info(info_bots, bstrinfo_bots, keys_bstr)
    info_test = append_info(info_test, bstrinfo_test, keys_bstr)

    info_humans.fillna(0, inplace=True)
    info_bots.fillna(0, inplace=True)
    info_test.fillna(0, inplace=True)
    ##########################################################################
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

    ##########################################################################
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

    ##########################################################################
    # Outlier dropping
    ##########################################################################
    print "Removing outliers..."

    bots_outliers = [
        '7fab82fa5eaea6a44eb743bc4bf356b3tarle',
        'f35082c6d72f1f1be3dd23f949db1f577t6wd',
        'bd0071b98d9479130e5c053a244fe6f1muj8h',
        '91c749114e26abdb9a4536169f9b4580huern',
        '74a35c4376559c911fdb5e9cfb78c5e4btqew'
    ]
    info_bots.drop(bots_outliers, inplace=True)

    ##########################################################################
    # Feature selection
    ##########################################################################
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

    print "Extracting keys..."
    info_humans = info_humans[keys_use]
    info_bots = info_bots[keys_use]
    info_test = info_test[keys_use]

    print "Saving prprocessed data.."
    info_humans.to_csv('data/info_humans_pp1.csv')
    info_bots.to_csv('data/info_bots_pp1.csv')
    info_test.to_csv('data/info_test_pp1.csv')

elif 'resume' in argv[1]:
    ##########################################################################
    # Save/Load preprocessed data
    ##########################################################################

    if len(argv)>2:
        n_resume = float(argv[2])
    else:
        n_resume = 9
    
    print "loading preprocessed data..."
    info_humans = pd.read_csv('data/info_humans_pp%d.csv' %(n_resume),
                              index_col=0)
    info_bots = pd.read_csv('data/info_bots_pp%d.csv' %(n_resume),
                            index_col=0)
    info_test = pd.read_csv('data/info_test_pp%d.csv' %(n_resume),
                            index_col=0)

    print info_test.describe()
else:
    sys.exit(1)

############################################################################
# k-fold Cross Validaton
############################################################################
# params for xgb
params_xgb = {'model': 'XGB', 'colsample_bytree': 0.367, 'silent': 1,
              'num_rounds': 1000, 'nthread': 8, 'min_child_weight': 3.0,
              'subsample': 0.9, 'eta': 0.002, 'max_depth': 5.0, 'gamma': 1.0}

# params for et
params_et = {'model': 'ET', 'n_estimators': 3000, 'max_features': 'auto',
            'criterion': 'gini', 'plot_importance': False, 'verbose': 0,
            'n_jobs': 2}

# params for RF
params_rf = {'model': 'RF', 'n_estimators': 1000, 'max_features': 'auto',
             'criterion': 'gini', 'plot_importance': False, 'verbose': 0,
             'n_jobs': -1, 'max_depth': 3}

# params for logistic regression
# params = {'model': 'logistic'}

# params for svc
params_svc = {'model': 'SVC', 'C': 1.0, 'gamma': 1.0}

# params for kneighbor
params_kn = {'model': 'KN', 'n_neighbors': 256, 'weights': 'distance',
             'algorithm': 'auto'}

params_ens = [params_xgb, params_et, params_svc, params_rf, params_kn]
params_ens = [params_xgb, params_et]

roc_aucs = fb_funcs.kfcv_ens(info_humans, info_bots, params_ens,
                             num_cv=5, num_folds=5)


roc_aucs = pd.DataFrame(np.array(roc_aucs), index=['auc', 'std'],
                        columns = ['xgb', 'et', 'ens'])
                        # columns=['xgb', 'et', 'svc', 'rf', 'kn', 'ens'])
roc_aucs.to_csv('data/submi/roc_aucs.csv', float_format='%11.6f')

print "cross validation results:"
print roc_aucs

############################################################################
# fit and predict
############################################################################

y_test_proba, ytps\
    = fb_funcs.fit_and_predict(info_humans, info_bots, info_test,
                               params=params_ens)

############################################################################
# submission file generation
############################################################################
submissionfile = 'data/submi/sub_ens.csv'
testfile = 'data/test.csv'

print "writing a submission file..."
write_submission(y_test_proba, info_test.index, testfile, submissionfile)

print "writing results from different models..."
for i in range(len(ytps)):
    submf = 'data/submi/sub_%s.csv' %(params_ens[i]['model'])
    write_submission(ytps[i], info_test.index, testfile, submf)
