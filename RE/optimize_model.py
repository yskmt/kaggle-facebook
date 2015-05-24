"""
optimize_model.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import json
import numpy as np
import pandas as pd

from sklearn.metrics import auc

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb

from fb_funcs import (append_merchandise, predict_cv,
                      fit_and_predict, append_info,
                      append_countries, keys_sig, keys_na,
                      append_bba, append_device, append_bids_intervals)

############################################################################
# Load bsic data
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

# cts_appended = keys_sig + keys_na
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
# Numer of same-time-bids data
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

keys_basic = ['num_bids', 'num_aucs', 'num_ips', 'num_devices',
              'num_urls', 'num_countries', 'num_merchs']

keys_merchandises = ['computers', 'office equipment', 'auto parts', 'sporting goods',
                     'books and music', 'clothing', 'furniture', 'jewelry', 'mobile',
                     'home goods']

keys_countries = ['ch', 'cn', 'ca', 'za', 'ec', 'ar', 'au', 'in', 'my', 'ru',
                  'nl', 'no', 'tw', 'id', 'lv', 'lt', 'lu', 'th', 'fr', 'jp', 'bn',
                  'de', 'bh', 'it', 'br', 'ph', 'sg', 'us', 'qa', 'kr', 'uk', 'bf',
                  'sa', 'ua']

keys_devices = ['phone136', 'phone640', 'phone739', 'phone150', 'phone15',
                'phone33', 'phone1030', 'phone996', 'phone58', 'phone55',
                'phone2287', 'phone205', 'phone224', 'phone90', 'phone3359',
                'phone143', 'phone168', 'phone144', 'phone728', 'phone6',
                'phone2330', 'phone28', 'phone25', 'phone1026', 'phone21',
                'phone239', 'phone22', 'phone219', 'phone195', 'phone46', 'phone63',
                'phone65', 'phone110', 'phone469']

keys_bbaucs = ['bba_35', 'bba_33', 'bba_32', 'bba_31', 'bba_30', 'bba_19',
               'bba_18', 'bba_15', 'bba_14', 'bba_17', 'bba_16', 'bba_11', 'bba_10',
               'bba_13', 'bba_12', 'bba_28', 'bba_29', 'bba_20', 'bba_21', 'bba_22',
               'bba_23', 'bba_24', 'bba_25', 'bba_26', 'bba_27', 'bba_9', 'bba_8',
               'bba_5', 'bba_4', 'bba_7', 'bba_6', 'bba_1', 'bba_3', 'bba_2']

keys_bintervals = ['int_0', 'int_1', 'int_2', 'int_3', 'int_4',
                   'int_5', 'int_6', 'int_7', 'int_8', 'int_9', 'int_10']

keys_nbs = ['num_bids_sametime_sameauc', 'num_bids_sametime_diffauc']

keys_bstr = ['streak_0', 'streak_1', 'streak_2', 'streak_3',
             'streak_4', 'streak_5', 'streak_6', 'streak_7', 'streak_8', 'streak_9']

# keys_use = keys_basic[:-1] + keys_merchandises + keys_countries + \
#     keys_devices + keys_bbaucs + keys_bintervals + keys_nbs + keys_bstr

keys_use = keys_all

print "Extracting keys..."
info_humans = info_humans[keys_use]
info_bots = info_bots[keys_use]
info_test = info_test[keys_use]


def xgb_objective(params):
    print params

    num_humans = len(info_humans)
    num_bots = len(info_bots)

    # combine humans and bots data to create given data
    info_given = pd.concat([info_humans, info_bots], axis=0)
    labels_train = np.hstack((np.zeros(num_humans), np.ones(num_bots)))
    num_given = len(labels_train)

    # shuffle just in case
    index_sh = np.random.choice(num_given, num_given, replace=False)
    info_given = info_given.iloc[index_sh]
    labels_train = labels_train[index_sh]

    # get matrices forms
    X_train = info_given.sort(axis=1).as_matrix()
    y_train = labels_train
    X_test = info_test.sort(axis=1).as_matrix()

    features = info_given.sort(axis=1).keys()

    # xgboost!
    xgb_params = {"objective": "binary:logistic", 'eta':
                  params['eta'], 'gamma': params['gamma'],
                  'max_depth': params['max_depth'], 'subsample': params['subsample'],
                  'colsample_bytree': params['colsample_bytree']}
    num_rounds = int(params['num_rounds'])

    dtrain = xgb.DMatrix(X_train, label=y_train)

    cv = 5
    cv_result = xgb.cv(xgb_params, dtrain, num_rounds, nfold=cv,
                       metrics={'rmse', 'error', 'auc'}, seed=0)

    auc_max = np.max(np.array(
        map(lambda x: float(x.split('\t')[1].split(':')[1].split('+')[0]), cv_result)))
    ind_max = np.argmax(np.array(
        map(lambda x: float(x.split('\t')[1].split(':')[1].split('+')[0]), cv_result)))


    # logging
    with open('log_results.txt', 'a') as f:
        f.write(str({'loss': -auc_max, 'status': STATUS_OK, 'ind': ind_max}))
        f.write('\n')
    with open('log_params.txt', 'a') as f:
        f.write(str(params))
        f.write('\n')
    
    return {'loss': -auc_max, 'status': STATUS_OK, 'ind': ind_max}


def optimize(trials):
    space = {
        'num_rounds': 5000,
        'eta': hp.quniform('eta', 0.001, 0.5, 0.001),
        'max_depth': hp.quniform('max_depth', 4, 13, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.001, 1, 0.001),
        'nthread': 6,
        'silent': 1
    }

    best = fmin(xgb_objective, space,
                algo=tpe.suggest, trials=trials, max_evals=100)

    # logging
    with open('trials_results.txt', 'w') as f:
        json.dump(trials.results, f)
    with open('trials_trials.txt', 'w') as f:
        json.dump(trials.trials, f)

    print best


# Trials object where the history of search will be stored
trials = Trials()

best = optimize(trials)

print best
