"""
predict.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

from pdb import set_trace
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

from fb_funcs import (append_merchandise, predict_cv,
                      fit_and_predict, append_info,
                      append_countries, keys_sig, keys_na,
                      append_bba, append_device, append_bids_intervals)
from feature_selection import select_k_best_features


start_time = time.time()

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
# bid counts for each period
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

print "Extracting keys..."
info_humans = info_humans[keys_use]
info_bots = info_bots[keys_use]
info_test = info_test[keys_use]


############################################################################
# Save/Load preprocessed data
############################################################################

# info_humans.to_csv('data/info_humans_pp2.csv')
# info_bots.to_csv('data/info_bots_pp2.csv')
# info_test.to_csv('data/info_test_pp2.csv')

# info_humans = read_csv(info_humans.to_csv('data/info_humans_pp.csv')
# info_bots = read_csv('data/info_bots_pp.csv')
# info_test = read_csv('data/info_test_pp.csv')

############################################################################
# k-fold Cross Validaton
############################################################################
# print "K-fold CV..."

# roc_auc = []
# roc_auc_std = []
# clf_score = []

# num_cv = 5
# for i in range(num_cv):
#     clf, ra, cs \
#         = predict_cv(info_humans, info_bots, n_folds=5,
#                      n_estimators=2000, plot_roc=False, model='XGB')

#     print ra.mean(), ra.std()
#     # print cs.mean(), cs.std()

#     roc_auc.append(ra.mean())
#     roc_auc_std.append(ra.std())
#     # clf_score.append(cs.mean())

# roc_auc = np.array(roc_auc)
# roc_auc_std = np.array(roc_auc_std)
# # clf_score = np.array(clf_score)

# print ""
# print roc_auc.mean(), roc_auc_std.mean()
# # print clf_score.mean(), clf_score.std()


############################################################################
# fit and predict
############################################################################

y_test_proba, y_train_proba, cvr, feature_importance\
    = fit_and_predict(info_humans, info_bots, info_test, model='XGB_CV',
                      n_estimators=2000, p_use=None, plot_importance=False)

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

# first method
submission = pd.DataFrame(
    y_test_proba, index=info_test.index, columns=['prediction'])
test_bidders = pd.read_csv('data/test.csv', index_col=0)

submission = pd.concat([submission, test_bidders], axis=1)
submission.fillna(0, inplace=True)
submission.to_csv('data/submission.csv', columns=['prediction'],
                  index_label='bidder_id')


