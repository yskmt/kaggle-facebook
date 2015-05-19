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
                      append_bba)


start_time = time.time()
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

# if using the merchandise data, dummies needs to be created

info_humans = append_merchandise(info_humans, drop=True)
info_bots = append_merchandise(info_bots, drop=True)
info_test = append_merchandise(info_test, drop=True)

# add country counts that can be significant
# load countrty info from file
cinfo_humans = pd.read_csv('data/country_info_humans.csv', index_col=0)
cinfo_bots = pd.read_csv('data/country_info_bots.csv', index_col=0)
cinfo_test = pd.read_csv('data/country_info_test.csv', index_col=0)

cts_appended = keys_sig+keys_na

info_humans = append_countries(info_humans, cinfo_humans, cts_appended)
info_bots = append_countries(info_bots, cinfo_bots, cts_appended)
info_test = append_countries(info_test, cinfo_test, cts_appended)

info_humans.fillna(0, inplace=True)
info_bots.fillna(0, inplace=True)
info_test.fillna(0, inplace=True)

# bids-by-auction data
# load bids-by-auction data from file
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
# Outlier dropping
############################################################################

bots_outliers = [
    '7fab82fa5eaea6a44eb743bc4bf356b3tarle',
    'f35082c6d72f1f1be3dd23f949db1f577t6wd',
    'bd0071b98d9479130e5c053a244fe6f1muj8h',
    '91c749114e26abdb9a4536169f9b4580huern',
    '74a35c4376559c911fdb5e9cfb78c5e4btqew'
]
info_bots.drop(bots_outliers, inplace=True)

############################################################################
# Data dropping
############################################################################

keys_all = info_humans.keys()
# [u'num_bids', u'num_aucs', u'num_merchs', u'num_devices',
# u'num_countries', u'num_ips', u'num_urls', u'merchandise'],

# decide which keys to use
if 'merchandise' in keys_all:
    keys_use = keys_all.drop(['merchandise'])
else:
    keys_use = keys_all

# 40 features
# keys_use = [u'au', u'num_bids', u'bba_1', u'id', u'bba_4', u'th',
#             u'bba_5', u'num_devices', u'bba_2', u'bba_3', u'num_urls', u'ar',
#             u'bba_9', u'bba_6', u'bba_8', u'bba_7', u'num_ips', u'bba_10',
#             u'bba_11', u'bba_12', u'num_aucs', u'num_countries', u'nl', u'bba_14',
#             u'bba_15', u'bba_16', u'bba_13', u'bba_17', u'bba_20', u'bba_21',
#             u'sporting goods', u'bba_19', u'bba_18', u'bba_23', u'bba_22',
#             u'bba_24', u'bba_25', u'bba_29', u'bba_27', u'mobile']

# 22 features
keys_use = [u'au', u'id', u'num_bids', u'bba_1', u'bba_4', u'th',
            u'bba_5', u'num_devices', u'bba_2', u'bba_3', u'num_urls',
            u'bba_6', u'bba_9', u'ar', u'bba_8', u'bba_7', u'bba_10',
            u'bba_11', u'num_ips', u'bba_12', u'num_aucs',
            u'num_countries']
# keys_use = keys_use[:10]
    
# keys_use = ['num_bids', 'num_aucs', 'num_countries', 'num_ips', 'num_urls']

# , u'num_aucs',u'num_devices',
            # u'num_countries', u'num_ips', u'num_urls']
    
# drop keys
print "dropping some keys..."
print "The keys to use: ", list(keys_use)
for key in keys_all:
    if key not in keys_use:
        info_humans.drop(key, axis=1, inplace=True)
        info_bots.drop(key, axis=1, inplace=True)
        info_test.drop(key, axis=1, inplace=True)


############################################################################
# k-fold Cross Validaton
############################################################################

roc_auc = []
roc_auc_std = []
clf_score = []

num_cv = 5
for i in range(num_cv):
    clf, ra, cs, tpr_50 \
        = predict_cv(info_humans, info_bots, n_folds=5,
                     n_estimators=1000, plot_roc=False)
    
    print ra.mean(), ra.std()
    print cs.mean(), cs.std()
    # print tpr_50.mean(), tpr_50.std()
    
    roc_auc.append(ra.mean())
    roc_auc_std.append(ra.std())
    clf_score.append(cs.mean())

roc_auc = np.array(roc_auc)
roc_auc_std = np.array(roc_auc_std)
clf_score = np.array(clf_score)

print ""
print roc_auc.mean(), roc_auc_std.mean()
print clf_score.mean(), clf_score.std()
# print tpr_50


############################################################################
# fit and predict
############################################################################

y_test_proba, y_train_proba, _\
    = fit_and_predict(info_humans, info_bots, info_test, model='ET',
                      n_estimators=1000, p_use=None, plotting=False)

############################################################################
# xgboost: CV
############################################################################

# y_pred, ytrain_pred, cv_result \
#     = fit_and_predict(info_humans, info_bots, info_test,
#                       n_estimators=20, p_use=None, cv=5)

# auc = []
# for i in range(len(cv_result)):
#     auc.append(float(cv_result[i].split('\t')[1].split(':')[1].split('+')[0]))

# best_itr = np.argmax(auc)
# auc_std = float(cv_result[11].split('\t')[1].split('+')[1])
# auc_best = np.max(auc)
# print "itr:", best_itr, "auc:", auc_best, "+=", auc_std


# ############################################################################
# # xgboost: prediction
# ############################################################################
# ytestp = []
# ytrainp = []

# for i in range(40):
#     y_test_proba, y_train_pred, y_train, cv_result \
#         = fit_and_predict(info_humans, info_bots, info_test,
#                           n_estimators=400, p_use=p_use)

#     ytestp.append(y_test_proba)
#     ytrainp.append(y_train_pred)


############################################################################
# submission file generation
############################################################################

# 70 bidders in test.csv do not have any data in bids.csv. Thus they
# are not included in analysis/prediction, but they need to be
# appended in the submission. The prediction of these bidders do not matter.

print "writing a submission file..."

# first method
submission = pd.DataFrame(y_test_proba, index=info_test.index, columns=['prediction'])
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
#                                  index=test_ids_append, columns=['prediction'])

# # Make as submission file!
# submission = pd.DataFrame(y_test_proba, index=test_ids,
#                           columns=['prediction'])
# submission = pd.concat([submission, submission_append], axis=0)
# submission.to_csv('data/submission.csv', index_label='bidder_id')

# end_time = time.time()
# print "Time elapsed: %.2f" % (end_time - start_time)


############################################################################
# k-fold Cross Validation with Bagging
############################################################################

# # under/over sampleing
# num_humans_use = num_humans
# num_bots_use = num_bots

# p_valid = 0.2
# num_sim = 20

# auc_score = []
# accuracy = []
# tpr = []
# for i in range(num_sim):
#     print "%d/%d" %(i, num_sim)

#     clf, roc_auc, clf_score, tpr_50\
#         = predict_usample(num_humans, num_humans_use, num_bots_use,
#                           info_humans, info_bots, plot_roc=False,
#                           p_valid=p_valid)

#     auc_score.append(roc_auc)
#     accuracy.append(clf_score)
#     tpr.append(tpr_50)

# auc_score = np.array(auc_score)
# accuracy = np.array(accuracy)
# tpr = np.array(tpr)

# print "auc score:", auc_score.mean(), auc_score.std()
# print "tpr score:", tpr.mean(), tpr.std()
# print "accuracy:", accuracy.mean(), accuracy.std()
