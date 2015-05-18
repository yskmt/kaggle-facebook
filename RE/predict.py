"""
predict.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

from fb_funcs import (predict_usample, append_merchandise, predict_cv,
                      fit_and_predict,
                      append_countries, keys_sig, keys_na)


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

# info_humans = append_merchandise(info_humans, drop=True)
# info_bots = append_merchandise(info_bots, drop=True)
# info_test = append_merchandise(info_test, drop=True)

# add country counts that can be significant
# load countrty info from file
# cinfo_humans = pd.read_csv('data/country_info_humans.csv', index_col=0)
# cinfo_bots = pd.read_csv('data/country_info_bots.csv', index_col=0)
# cinfo_test = pd.read_csv('data/country_info_test.csv', index_col=0)

# info_humans = append_countries(info_humans, cinfo_humans, keys_sig+keys_na)
# info_bots = append_countries(info_bots, cinfo_bots, keys_sig+keys_na)
# info_test = append_countries(info_test, cinfo_test, keys_sig+keys_na)

# info_humans.fillna(0, inplace=True)
# info_bots.fillna(0, inplace=True)
# info_test.fillna(0, inplace=True)

############################################################################
# Data dropping
############################################################################

keys_all = info_humans.keys()
# [u'num_bids', u'num_aucs', u'num_merchs', u'num_devices',
# u'num_countries', u'num_ips', u'num_urls', u'merchandise'],

# decide which keys to use
keys_use = \
    keys_all.drop(['merchandise'])
# keys_use = keys_all

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

# roc_auc = 0.0
# clf_score = 0.0

# num_cv = 3
# for i in range(num_cv):
#     clf, ra, cs, tpr_50 \
#         = predict_cv(info_humans, info_bots, n_folds=3,
#                      n_estimators=500, plot_roc=True)

    
#     print ra.mean(), ra.std()
#     print cs.mean(), cs.std()
#     # print tpr_50.mean(), tpr_50.std()
    
#     roc_auc += ra.mean()
#     clf_score += cs.mean()

# print roc_auc/num_cv
# print clf_score/num_cv
# # print tpr_50


############################################################################
# fit and predict
############################################################################

# clf, y_test_proba = fit_and_predict(info_humans, info_bots, info_test,
#                                     n_estimators=10000, p_use=None)



############################################################################
# xgboost: CV
############################################################################
y_pred, yrain_pred, cv_result \
    = fit_and_predict(info_humans, info_bots, info_test,
                      n_estimators=100, p_use=None, cv=5)

auc = []
for i in range(len(cv_result)):
    auc.append(float(cv_result[i].split('\t')[1].split(':')[1].split('+')[0]))

best_itr = np.argmax(auc)
auc_std = float(cv_result[11].split('\t')[1].split('+')[1])
auc_best = np.max(auc)
print "itr:", best_itr, "auc:", auc_best, "+=", auc_std


############################################################################
# xgboost: prediction
############################################################################
y_test_proba, yrain_pred, cv_result \
    = fit_and_predict(info_humans, info_bots, info_test,
                      n_estimators=best_itr, p_use=None)

# 70 bidders in test.csv do not have any data in bids.csv. Thus they
# are not included in analysis/prediction, but they need to be
# appended in the submission. The prediction of these bidders do not matter.

test_ids = info_test.index
test_ids_all = pd.read_csv('data/test.csv')['bidder_id']
test_ids_append = list(
    set(test_ids_all.values).difference(set(test_ids.values)))
submission_append = pd.DataFrame(np.zeros(len(test_ids_append)),
                                 index=test_ids_append, columns=['prediction'])

# Make as submission file!
submission = pd.DataFrame(y_test_proba, index=test_ids,
                          columns=['prediction'])
submission = pd.concat([submission, submission_append], axis=0)
submission.to_csv('data/submission.csv', index_label='bidder_id')

end_time = time.time()
print "Time elapsed: %.2f" % (end_time - start_time)


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
