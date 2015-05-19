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
                      append_bba, append_device)
from feature_selection import select_k_best_features


start_time = time.time()

print "Loading preprocessed data files..."
info_humans = pd.read_csv('data/info_humans_pp.csv', index_col=0)
info_bots = pd.read_csv('data/info_bots_pp.csv', index_col=0)
info_test = pd.read_csv('data/info_test_pp.csv', index_col=0)



############################################################################
# Feature dropping
############################################################################
print "Dropping features..."
keys_all = info_humans.keys()
keys_use = ['bba_3', 'bba_2', 'num_bids', 'bba_1', 'num_ips', 'phone46', 'au',
            'phone143', 'phone28', 'phone13', 'th', 'phone17', 'phone290',
            'phone62', 'phone157', 'phone479', 'phone237', 'phone248', 'phone346',
            'phone119', 'phone56', 'phone122']
keys_use = keys_use[:22]

# drop keys
print "dropping some keys..."
print "The keys to use: \n", list(keys_use)
for key in keys_all:
    if key not in keys_use:
        info_humans.drop(key, axis=1, inplace=True)
        info_bots.drop(key, axis=1, inplace=True)
        info_test.drop(key, axis=1, inplace=True)

############################################################################
# k-fold Cross Validaton
############################################################################
print "K-fold CV..."

roc_auc = []
roc_auc_std = []
clf_score = []

num_cv = 5
for i in range(num_cv):
    clf, ra, cs, tpr_50 \
        = predict_cv(info_humans, info_bots, n_folds=5,
                     n_estimators=2000, plot_roc=False)

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
                      n_estimators=1000, p_use=None, plotting=True)

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
