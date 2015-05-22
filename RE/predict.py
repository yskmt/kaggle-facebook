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

from fb_funcs import (predict_usample, append_merchandise, predict_cv,
                      fit_and_predict,
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

# features selection by chi2 test
# num_features = 40
# indx_ex, ft_ex = select_k_best_features(num_features, info_humans, info_bots)
# keys_use = ft_ex

# 40 out of 7495 features globally
# by chi2 test
# keys_use = ['au', 'bba_1', 'bba_2', 'bba_3', 'num_bids', 'num_ips',
#             'phone1029', 'phone113', 'phone115', 'phone118',
#             'phone119', 'phone1211', 'phone122', 'phone13',
#             'phone143', 'phone157', 'phone17', 'phone201', 'phone204',
#             'phone237', 'phone248', 'phone278', 'phone28', 'phone290',
#             'phone322', 'phone346', 'phone386', 'phone389',
#             'phone391', 'phone46', 'phone466', 'phone479', 'phone503',
#             'phone524', 'phone528', 'phone56', 'phone62', 'phone718',
#             'phone796', 'th']

# # by using extratrees clf globally
# keys_use = ['bba_5', 'bba_4', 'bba_3', 'phone46', 'num_bids', 'bba_6', 'bba_8',
#             'bba_2', 'bba_1', 'bba_7', 'au', 'bba_10', 'bba_9', 'bba_12',
#             'bba_14', 'phone195', 'bba_11', 'num_aucs', 'bba_13', 'phone143',
#             'bba_15', 'phone63', 'bba_16', 'num_ips', 'bba_20', 'bba_17',
#             'bba_18', 'phone55', 'num_urls', 'phone1030', 'num_countries',
#             'phone150', 'phone144', 'bba_21', 'num_devices', 'phone33', 'bba_19',
#             'bba_22', 'phone1026', 'bba_24']
# keys_use = keys_use[:31]

# # # 10*4 features selected from each category by chi2
# keys_use1 = ['phone115', 'phone119', 'phone122', 'phone13', 'phone17',
#             'phone237', 'phone389', 'phone46', 'phone62', 'phone718',
#             'at', 'au', 'ca', 'de', 'in', 'jp', 'kr', 'ru', 'th',
#             'us', 'bba_1', 'bba_14', 'bba_2', 'bba_3', 'bba_4',
#             'bba_5', 'bba_6', 'bba_7', 'bba_8', 'bba_9', 'computers',
#             'jewelry', 'mobile', 'num_aucs', 'num_bids',
#             'num_countries', 'num_devices', 'num_ips', 'num_urls',
#             'sporting goods']

# # 10*4 features selected from each category by ET
# keys_use2 = ['au', 'num_bids', 'za', 'phone55', 'phone739',
#             'num_devices', 'ca', 'my', 'num_ips', 'num_aucs',
#             'num_urls', 'phone996', 'phone150', 'phone640', 'bba_14',
#             'bba_15', 'num_countries', 'phone136', 'in', 'phone33',
#             'cn', 'bba_17', 'ch', 'ru', 'ar', 'bba_19', 'bba_18',
#             'phone58', 'bba_30', 'phone1030', 'bba_31', 'bba_33',
#             'phone15', 'bba_32', 'bba_35', 'ec']

# keys_use = list(set(keys_use1).union(keys_use2))


keys_devices = ['phone136', 'phone640', 'phone739', 'phone150', 'phone15',
                'phone33', 'phone1030', 'phone996', 'phone58', 'phone55',
                'phone2287', 'phone205', 'phone224', 'phone90', 'phone3359',
                'phone143', 'phone168', 'phone144', 'phone728', 'phone6',
                'phone2330', 'phone28', 'phone25', 'phone1026', 'phone21',
                'phone239', 'phone22', 'phone219', 'phone195', 'phone46', 'phone63',
                'phone65', 'phone110', 'phone469']

keys_countries = ['ch', 'cn', 'ca', 'za', 'ec', 'ar', 'au', 'in', 'my', 'ru',
                  'nl', 'no', 'tw', 'id', 'lv', 'lt', 'lu', 'th', 'fr', 'jp', 'bn',
                  'de', 'bh', 'it', 'br', 'ph', 'sg', 'us', 'qa', 'kr', 'uk', 'bf',
                  'sa', 'ua']

keys_bbaucs = ['bba_35', 'bba_33', 'bba_32', 'bba_31', 'bba_30', 'bba_19',
               'bba_18', 'bba_15', 'bba_14', 'bba_17', 'bba_16', 'bba_11', 'bba_10',
               'bba_13', 'bba_12', 'bba_28', 'bba_29', 'bba_20', 'bba_21', 'bba_22',
               'bba_23', 'bba_24', 'bba_25', 'bba_26', 'bba_27', 'bba_9', 'bba_8',
               'bba_5', 'bba_4', 'bba_7', 'bba_6', 'bba_1', 'bba_3', 'bba_2']

keys_merchandises = ['computers', 'office equipment', 'auto parts', 'sporting goods',
                     'books and music', 'clothing', 'furniture', 'jewelry', 'mobile',
                     'home goods']

keys_bintervals = ['2', '8', '1', '10', '3', '7', '4', '9', '58', '6', '0',
                   '11', '12', '20', '5', '18', '19', '13', '27', '21', '15',
                   '22', '16', '17', '26', '46', '25', '64', '30', '14', '52',
                   '23', '61', '29', '36', '32', '34', '50', '24', '38', '67',
                   '39', '54', '31', '35', '53', '57', '62', '37', '60', '33',
                   '43', '28', '81', '55', '70', '76', '47', '72', '40', '49',
                   '79', '93', '66', '78', '56', '82', '86', '74', '45', '41',
                   '48', '59', '87', '68', '92', '85', '73', '44', '91', '94',
                   '42', '97', '63', '77', '71', '98', '84', '90', '88', '80',
                   '83', '75', '99', '65', '96', '51', '89', '95', '69']

keys_bintervals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

keys_counts = ['num_bids', 'num_aucs', 'num_ips', 'num_devices',
               'num_urls', 'num_countries', 'num_merchs']

keys_use = keys_counts[:-1] + keys_devices + keys_countries + keys_bbaucs\
           + keys_merchandises + keys_bintervals

keys_use = keys_all
# keys_use = keys_use[:30]


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
#     clf, ra, cs, tpr_50 \
#         = predict_cv(info_humans, info_bots, n_folds=5,
#                      n_estimators=10000, plot_roc=False)

#     print ra.mean(), ra.std()
#     print cs.mean(), cs.std()
#     # print tpr_50.mean(), tpr_50.std()

#     roc_auc.append(ra.mean())
#     roc_auc_std.append(ra.std())
#     clf_score.append(cs.mean())

# roc_auc = np.array(roc_auc)
# roc_auc_std = np.array(roc_auc_std)
# clf_score = np.array(clf_score)

# print ""
# print roc_auc.mean(), roc_auc_std.mean()
# print clf_score.mean(), clf_score.std()
# # print tpr_50


############################################################################
# fit and predict
############################################################################

y_test_proba, y_train_proba, _, feature_importance\
    = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                      n_estimators=10000, p_use=None, plot_importance=True)

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


# #  second method
# test_ids = info_test.index
# test_ids_all = pd.read_csv('data/test.csv')['bidder_id']
# test_ids_append = list(
#     set(test_ids_all.values).difference(set(test_ids.values)))
# submission_append = pd.DataFrame(np.zeros(len(test_ids_append)),
# index=test_ids_append, columns=['prediction'])

# # Make as submission file!
# submission = pd.DataFrame(y_test_proba, index=test_ids,
#                           columns=['prediction'])
# submission = pd.concat([submission, submission_append], axis=0)
# submission.to_csv('data/submission.csv', index_label='bidder_id')

# end_time = time.time()
# print "Time elapsed: %.2f" % (end_time - start_time)
